from pathlib import Path
from urllib.parse import urlparse
import re

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "phiusiil+phishing+url+dataset.csv"
ARTIFACT_PATH = BASE_DIR / "models" / "url_phishing_pipeline.joblib"
TARGET_COLUMN = "label"

# URL features that can be computed directly from a raw URL string.
URL_FEATURE_COLUMNS = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
]

app = Flask(__name__)


def _count_subdomains(host: str) -> int:
    parts = [p for p in host.split(".") if p]
    return max(0, len(parts) - 2)


def _is_ip_address(host: str) -> int:
    pattern = r"^\d{1,3}(\.\d{1,3}){3}$"
    return int(bool(re.match(pattern, host)))


def extract_features_from_url(raw_url: str) -> dict:
    url = raw_url.strip()
    if not url:
        raise ValueError("URL is required.")
    if "://" not in url:
        url = f"http://{url}"

    parsed = urlparse(url)
    host = parsed.netloc.split(":")[0]
    path_and_query = f"{parsed.path or ''}{('?' + parsed.query) if parsed.query else ''}"

    url_length = len(url)
    letters = sum(ch.isalpha() for ch in url)
    digits = sum(ch.isdigit() for ch in url)
    equals = url.count("=")
    qmarks = url.count("?")
    ampersands = url.count("&")
    obfuscated = url.count("%")
    special_chars = len(re.findall(r"[^a-zA-Z0-9]", url))

    return {
        "URLLength": float(url_length),
        "DomainLength": float(len(host)),
        "IsDomainIP": float(_is_ip_address(host)),
        "NoOfSubDomain": float(_count_subdomains(host)),
        "HasObfuscation": float(int("%" in url or "@" in url)),
        "NoOfObfuscatedChar": float(obfuscated),
        "ObfuscationRatio": float(obfuscated / url_length if url_length else 0.0),
        "NoOfLettersInURL": float(letters),
        "LetterRatioInURL": float(letters / url_length if url_length else 0.0),
        "NoOfDegitsInURL": float(digits),
        "DegitRatioInURL": float(digits / url_length if url_length else 0.0),
        "NoOfEqualsInURL": float(equals),
        "NoOfQMarkInURL": float(qmarks),
        "NoOfAmpersandInURL": float(ampersands),
        "NoOfOtherSpecialCharsInURL": float(special_chars),
        "SpacialCharRatioInURL": float(special_chars / url_length if url_length else 0.0),
        "IsHTTPS": float(int(parsed.scheme.lower() == "https")),
        "_normalized_url": url,
        "_url_length_for_ui": len(path_and_query) + len(host) + len(parsed.scheme) + 3,
    }


def train_and_save_pipeline():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    missing_cols = [c for c in URL_FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    train_df = df[URL_FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    train_df = train_df.fillna(train_df.median(numeric_only=True))
    X = train_df[URL_FEATURE_COLUMNS].astype(float)
    y = train_df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)
    test_accuracy = float(model.score(X_test_scaled, y_test))

    # Build dashboard data from the provided dataset.
    url_length_hist = np.histogram(
        X["URLLength"].astype(float), bins=12
    )
    feature_importance_df = (
        pd.DataFrame(
            {
                "feature": URL_FEATURE_COLUMNS,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(10)
    )
    scatter_sample = (
        X[["URLLength", "NoOfOtherSpecialCharsInURL"]]
        .sample(n=min(350, len(X)), random_state=42)
        .astype(float)
    )

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": URL_FEATURE_COLUMNS,
            "test_accuracy": test_accuracy,
            "dashboard": {
                "url_length_distribution": {
                    "bin_edges": url_length_hist[1].tolist(),
                    "counts": url_length_hist[0].tolist(),
                },
                "feature_importance": feature_importance_df.to_dict(orient="records"),
                "scatter_points": scatter_sample.to_dict(orient="records"),
            },
        },
        ARTIFACT_PATH,
    )


def load_pipeline():
    if not ARTIFACT_PATH.exists():
        train_and_save_pipeline()
    return joblib.load(ARTIFACT_PATH)


@app.route("/", methods=["GET"])
def home():
    pipeline = load_pipeline()
    return render_template(
        "index.html",
        test_accuracy=pipeline["test_accuracy"],
        dashboard=pipeline.get("dashboard", {}),
    )


@app.route("/analyze", methods=["POST"])
def analyze_url():
    try:
        payload = request.get_json(silent=True) or {}
        raw_url = payload.get("url") or request.form.get("url", "")
        extracted = extract_features_from_url(raw_url)

        pipeline = load_pipeline()
        row = np.array([extracted[f] for f in pipeline["features"]], dtype=float).reshape(1, -1)
        row_scaled = pipeline["scaler"].transform(row)

        pred = int(pipeline["model"].predict(row_scaled)[0])
        proba = float(pipeline["model"].predict_proba(row_scaled)[0][1])
        special_char_count = int(extracted["NoOfOtherSpecialCharsInURL"])
        if proba >= 0.75:
            risk_level = "High"
        elif proba >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return jsonify(
            {
                "success": True,
                "url": extracted["_normalized_url"],
                "prediction": "Phishing" if pred == 1 else "Legitimate",
                "confidence": round(proba * 100, 2),
                "risk_level": risk_level,
                "url_length": int(extracted["_url_length_for_ui"]),
                "special_characters": special_char_count,
                "model_test_accuracy": round(pipeline["test_accuracy"] * 100, 2),
                "url_features": {
                    key: round(float(value), 4)
                    for key, value in extracted.items()
                    if not key.startswith("_")
                },
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/retrain", methods=["POST"])
def retrain():
    train_and_save_pipeline()
    pipeline = load_pipeline()
    return jsonify(
        {
            "success": True,
            "message": "Model retrained successfully.",
            "model_test_accuracy": round(pipeline["test_accuracy"] * 100, 2),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
