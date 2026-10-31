import os
import re
import math
import time
import joblib
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Suspicious TLDs frequently abused in phishing campaigns
SUSPICIOUS_TLDS = {
    'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'ru', 'work', 
    'buzz', 'monster', 'fit', 'kim', 'cc', 'su', 'icu', 'click'
}

# High-risk sensitive keywords commonly found in phishing lures
SUSPICIOUS_KEYWORDS = [
    'login', 'signin', 'verify', 'verification', 'account', 'banking',
    'secure', 'security', 'update', 'wallet', 'paypal', 'apple',
    'support', 'service', 'confirm', 'ebay', 'auth', 'recover', 'billing',
    'password', 'credential', 'webscr', 'alert', 'validate', 'chase', 'netflix'
]

IP_REGEX = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')

def calculate_entropy(text: str) -> float:
    """Calculates Shannon Entropy of a string."""
    if not text:
        return 0.0
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    return -sum(p * math.log2(p) for p in prob)

def extract_features_from_url(url: str) -> dict:
    """
    Extracts numerical and lexical features from a URL string.
    Returns a dictionary of feature names to values.
    """
    url_str = str(url).strip()
    
    # Ensure scheme for proper urlparse parsing
    parsed = urlparse(url_str if '://' in url_str else 'http://' + url_str)
    
    raw_host = (parsed.hostname or '').lower()
    # Strip single leading www.
    host = raw_host[4:] if raw_host.startswith('www.') else raw_host
    path = parsed.path or ''
    query = parsed.query or ''
    full_url = url_str.lower()
    
    # Hostname & URL lengths
    url_len = len(url_str)
    host_len = len(host)
    
    # Counts in host
    dots_host = host.count('.')
    hyphens_host = host.count('-')
    subdomain_count = max(0, dots_host - 1) if dots_host > 0 else 0
    digits_host = sum(c.isdigit() for c in host)
    digits_ratio_host = (digits_host / host_len) if host_len > 0 else 0.0
    entropy_host = calculate_entropy(host)
    
    # Counts in full URL
    count_slashes = url_str.count('/')
    count_special = sum(url_str.count(c) for c in ['_', '~', '%', '=', '&', '?', ';', '+', '$'])
    has_at_symbol = 1 if '@' in url_str else 0
    has_port = 1 if parsed.port is not None else 0
    is_https = 1 if url_str.lower().startswith('https://') else 0
    
    # IP address hostname check
    is_ip = 1 if IP_REGEX.match(raw_host) else 0
    
    # TLD check
    tld = host.split('.')[-1] if '.' in host else ''
    is_suspicious_tld = 1 if tld in SUSPICIOUS_TLDS else 0
    
    # Keyword analysis
    keyword_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in full_url)
    
    # Double slash redirect check (// appearing after initial http(s)://)
    url_after_scheme = url_str.split('://', 1)[-1] if '://' in url_str else url_str
    has_redirect = 1 if '//' in url_after_scheme else 0
    
    # Encoded characters check (%xx)
    has_hex = 1 if re.search(r'%[0-9a-fA-F]{2}', url_str) else 0

    return {
        'url_len': url_len,
        'host_len': host_len,
        'dots_host': dots_host,
        'hyphens_host': hyphens_host,
        'subdomain_count': subdomain_count,
        'digits_host': digits_host,
        'digits_ratio_host': digits_ratio_host,
        'entropy_host': entropy_host,
        'count_slashes': count_slashes,
        'count_special': count_special,
        'has_at_symbol': has_at_symbol,
        'has_port': has_port,
        'is_https': is_https,
        'is_ip': is_ip,
        'is_suspicious_tld': is_suspicious_tld,
        'keyword_count': keyword_count,
        'has_redirect': has_redirect,
        'has_hex': has_hex
    }

FEATURE_COLUMNS = [
    'url_len', 'host_len', 'dots_host', 'hyphens_host', 'subdomain_count',
    'digits_host', 'digits_ratio_host', 'entropy_host', 'count_slashes',
    'count_special', 'has_at_symbol', 'has_port', 'is_https', 'is_ip',
    'is_suspicious_tld', 'keyword_count', 'has_redirect', 'has_hex'
]

def load_data():
    """Finds and loads the phishing dataset."""
    candidates = [
        'data/phiusiil_cleaned.csv',
        'phiusiil_phishing_url_dataset_cleaned.csv'
    ]
    data_file = None
    for path in candidates:
        if os.path.exists(path):
            data_file = path
            break
            
    if not data_file:
        raise FileNotFoundError(f"Dataset not found. Checked: {candidates}")
        
    print(f"[*] Loading dataset from: {data_file}")
    df = pd.read_csv(data_file, usecols=['URL', 'label'])
    
    # Dataset label convention: 1 = Legitimate, 0 = Phishing
    # We map target: 1 = Phishing, 0 = Legitimate
    df['target'] = 1 - df['label']
    print(f"[*] Total rows: {len(df):,} | Phishing: {(df['target'] == 1).sum():,} | Legitimate: {(df['target'] == 0).sum():,}")
    return df

def train_and_evaluate(sample_size: int = None):
    start_time = time.time()
    df = load_data()
    
    if sample_size and len(df) > sample_size:
        print(f"[*] Sampling {sample_size:,} records for fast balanced training...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    print("[*] Extracting features from URLs...")
    feature_list = []
    urls = df['URL'].tolist()
    
    for i, u in enumerate(urls):
        feature_list.append(extract_features_from_url(u))
        if (i + 1) % 50000 == 0:
            print(f"    Processed {i + 1:,} / {len(urls):,} URLs...")
            
    X = pd.DataFrame(feature_list, columns=FEATURE_COLUMNS)
    y = df['target'].values
    
    print("[*] Splitting dataset (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("[*] Training Gradient Boosted Decision Tree Classifier...")
    model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    print("[*] Evaluating model performance...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("\n" + "=" * 50)
    print("        MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("=" * 50)
    print("\nDetailed Report:\n", classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    
    # Save model and metadata package
    model_artifact = {
        'model': model,
        'feature_columns': FEATURE_COLUMNS,
        'metrics': {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'roc_auc': float(roc_auc)
        },
        'trained_at': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    output_path = 'model.joblib'
    joblib.dump(model_artifact, output_path, compress=3)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] Saved model artifact to '{output_path}' ({file_size_mb:.2f} MB)")
    print(f"[OK] Total elapsed time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    train_and_evaluate()
