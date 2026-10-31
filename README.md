# 🛡️ PhishShield AI — Phishing Website Detection System

An intelligent, lightweight, and modern **Phishing Website Detection System** built with **Supervised Machine Learning** (Gradient Boosted Decision Trees) and **Flask**.

The system performs instant lexical, topological, and structural URL analysis to identify malicious links and phishing lures **without downloading or visiting the target URL**, ensuring fast, safe, and privacy-preserving detection.

---

## 🚀 Key Features

- **⚡ Instant Real-Time Analysis**: Evaluates URLs in milliseconds using lexical feature extraction (hostname entropy, sensitive keywords, IP obfuscation, subdomain depth, protocol validation).
- **🎯 High Accuracy**: Trained on **235,370 real-world URLs** with **99.8% accuracy** and **0.997 F1-score**.
- **🎨 Modern Dark Cyber-Security UI**: Visual risk score gauge (0–100%), threat breakdown cards, and quick-test demo chips.
- **🔌 RESTful API**: Includes `/predict` and `/health` endpoints for seamless integration into browser extensions, bots, or security pipelines.
- **📦 Clean & Lightweight**: Designed with minimal files (<2 MB total size including the pre-trained model artifact), making it **effortless to push and host on GitHub**.

---

## 📂 Project Structure

```text
PhishShield/
│── app.py             # Flask Web Server & Real-time Prediction Engine
│── train.py           # ML Model Training, Feature Engineering & Evaluation
│── model.joblib       # Pre-trained Compact ML Model (~145 KB)
│── requirements.txt   # Python Package Dependencies
│── .gitignore         # Keeps GitHub repo clean (ignores large CSVs & .venv)
│── README.md          # Documentation & Setup Guide
└── templates/
    └── index.html     # Responsive Cybersecurity Web Dashboard
```

---

## 🛠️ Quick Start Guide

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/phishing-detection-system.git
cd phishing-detection-system

# (Optional) Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to: **`http://localhost:5000`**

---

## 🧪 (Optional) Retrain the Machine Learning Model

If you have the dataset CSV (`data/phiusiil_cleaned.csv` or `phiusiil_phishing_url_dataset_cleaned.csv`), you can retrain the model at any time:

```bash
python train.py
```

### Model Performance Metrics
| Metric | Value |
| :--- | :--- |
| **Accuracy** | **99.80%** |
| **Precision** | **99.99%** |
| **Recall** | **99.55%** |
| **F1-Score** | **0.9977** |
| **ROC-AUC** | **0.9988** |
| **Model Size** | **~145 KB** |

---

## 📡 REST API Usage

### Predict URL Risk
**Endpoint:** `POST /predict`  
**Header:** `Content-Type: application/json`

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"url": "http://paypal-verification-alert.xyz/login"}'
```

**Example JSON Response:**
```json
{
  "url": "http://paypal-verification-alert.xyz/login",
  "host": "paypal-verification-alert.xyz",
  "is_phishing": true,
  "risk_score": 98.4,
  "risk_level": "Dangerous",
  "reasons": [
    "Domain uses high-risk Top Level Domain (.xyz) commonly seen in spam/phishing.",
    "Targeting sensitive keywords: login, paypal.",
    "Uses unencrypted HTTP instead of secure HTTPS."
  ],
  "features": {
    "is_https": 0,
    "is_ip": 0,
    "entropy_host": 3.74,
    "keyword_count": 2,
    "subdomain_count": 0
  }
}
```

### Health Check
**Endpoint:** `GET /healthz`
```bash
curl http://localhost:5000/healthz
```

---

## 📤 Uploading to GitHub

This repository is already configured with `.gitignore` so large dataset CSVs and virtual environments are excluded automatically:

```bash
git init
git add .
git commit -m "Initial commit: Lightweight Phishing Website Detection System"
git branch -M main
git remote add origin https://github.com/your-username/your-repository-name.git
git push -u origin main
```

---

## 📄 License
MIT License. Built for cybersecurity research and education.
