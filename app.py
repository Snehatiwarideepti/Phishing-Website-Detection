import os
import re
import math
import joblib
import pandas as pd
from urllib.parse import urlparse
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# List of trusted legitimate domains to eliminate false positives on verified services
ALLOWLIST = {
    'google.com', 'github.com', 'wikipedia.org', 'microsoft.com',
    'amazon.com', 'apple.com', 'youtube.com', 'linkedin.com',
    'twitter.com', 'x.com', 'reddit.com', 'stackoverflow.com',
    'facebook.com', 'instagram.com', 'netflix.com', 'yahoo.com',
    'cloudflare.com', 'gitlab.com', 'python.org', 'uci.edu'
}

SUSPICIOUS_TLDS = {
    'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'ru', 'work', 
    'buzz', 'monster', 'fit', 'kim', 'cc', 'su', 'icu', 'click'
}

SUSPICIOUS_KEYWORDS = [
    'login', 'signin', 'verify', 'verification', 'account', 'banking',
    'secure', 'security', 'update', 'wallet', 'paypal', 'apple',
    'support', 'service', 'confirm', 'ebay', 'auth', 'recover', 'billing',
    'password', 'credential', 'webscr', 'alert', 'validate', 'chase', 'netflix'
]

IP_REGEX = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')

FEATURE_COLUMNS = [
    'url_len', 'host_len', 'dots_host', 'hyphens_host', 'subdomain_count',
    'digits_host', 'digits_ratio_host', 'entropy_host', 'count_slashes',
    'count_special', 'has_at_symbol', 'has_port', 'is_https', 'is_ip',
    'is_suspicious_tld', 'keyword_count', 'has_redirect', 'has_hex'
]

# Load trained model package
MODEL_PATH = 'model.joblib'
model_artifact = None

def get_model():
    global model_artifact
    if model_artifact is None and os.path.exists(MODEL_PATH):
        try:
            model_artifact = joblib.load(MODEL_PATH)
            print(f"[OK] Loaded model trained at: {model_artifact.get('trained_at', 'unknown')}")
        except Exception as e:
            print(f"[!] Warning: Error loading model: {e}")
    return model_artifact

def calculate_entropy(text: str) -> float:
    if not text:
        return 0.0
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    return -sum(p * math.log2(p) for p in prob)

def normalize_host(url_str: str):
    """
    Normalizes URL and safely extracts host, handling user@host credentials,
    leading www, and ports.
    """
    clean = url_str.strip()
    parsed = urlparse(clean if '://' in clean else 'http://' + clean)
    raw_host = (parsed.hostname or '').lower()
    host = raw_host[4:] if raw_host.startswith('www.') else raw_host
    return parsed, host, raw_host

def is_domain_allowlisted(host: str) -> bool:
    if not host:
        return False
    for domain in ALLOWLIST:
        if host == domain or host.endswith('.' + domain):
            return True
    return False

def extract_features(url_str: str) -> tuple[dict, list[str], str]:
    parsed, host, raw_host = normalize_host(url_str)
    full_url = url_str.strip().lower()
    
    url_len = len(url_str)
    host_len = len(host)
    
    dots_host = host.count('.')
    hyphens_host = host.count('-')
    subdomain_count = max(0, dots_host - 1) if dots_host > 0 else 0
    digits_host = sum(c.isdigit() for c in host)
    digits_ratio_host = (digits_host / host_len) if host_len > 0 else 0.0
    entropy_host = calculate_entropy(host)
    
    count_slashes = url_str.count('/')
    count_special = sum(url_str.count(c) for c in ['_', '~', '%', '=', '&', '?', ';', '+', '$'])
    has_at_symbol = 1 if '@' in url_str else 0
    has_port = 1 if parsed.port is not None else 0
    is_https = 1 if url_str.strip().lower().startswith('https://') else 0
    is_ip = 1 if IP_REGEX.match(raw_host) else 0
    
    tld = host.split('.')[-1] if '.' in host else ''
    is_suspicious_tld = 1 if tld in SUSPICIOUS_TLDS else 0
    
    found_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in full_url]
    keyword_count = len(found_keywords)
    
    url_after_scheme = url_str.split('://', 1)[-1] if '://' in url_str else url_str
    has_redirect = 1 if '//' in url_after_scheme else 0
    has_hex = 1 if re.search(r'%[0-9a-fA-F]{2}', url_str) else 0
    
    # Generate human explanations
    reasons = []
    if is_ip:
        reasons.append("Raw IP address used as hostname instead of a standard domain name.")
    if has_at_symbol:
        reasons.append("URL contains '@' symbol (often used to obscure true destination host).")
    if is_suspicious_tld:
        reasons.append(f"Domain uses high-risk Top Level Domain (.{tld}) commonly seen in spam/phishing.")
    if found_keywords:
        reasons.append(f"Targeting sensitive keywords: {', '.join(found_keywords[:4])}.")
    if subdomain_count >= 3:
        reasons.append(f"High number of subdomain levels ({subdomain_count} subdomains).")
    if entropy_host > 3.8 and host_len > 12:
        reasons.append("High character entropy in hostname (suggests algorithmically generated domain).")
    if not is_https:
        reasons.append("Uses unencrypted HTTP instead of secure HTTPS.")
    if has_redirect:
        reasons.append("Contains double slash '//' in path, signaling potential open redirect.")

    # Display metrics for UI dashboard cards and charts
    digit_count = sum(c.isdigit() for c in url_str)
    letter_count = sum(c.isalpha() for c in url_str)
    has_http = 1 if url_str.strip().lower().startswith('http://') else 0
    special_char_count = sum(not c.isalnum() for c in url_str)
    
    display_features = {
        'DIGIT COUNT': digit_count,
        'DOMAIN LENGTH': host_len,
        'HAS AT': has_at_symbol,
        'HAS HTTP': has_http,
        'HAS HTTPS': is_https,
        'HAS IP': is_ip,
        'LETTER COUNT': letter_count,
        'SPECIAL CHAR COUNT': special_char_count,
        'SUBDOMAIN COUNT': subdomain_count,
        'URL LENGTH': url_len
    }
    
    feature_dict = {
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
    
    return feature_dict, reasons, host, display_features

def analyze_url(url: str) -> dict:
    if not url or not isinstance(url, str) or not url.strip():
        return {
            'error': 'Please provide a valid URL string.'
        }
        
    url = url.strip()
    feat_dict, reasons, host, display_features = extract_features(url)
    
    # 1. Allowlist check
    if is_domain_allowlisted(host):
        return {
            'url': url,
            'host': host,
            'is_phishing': False,
            'risk_score': 2.5,
            'risk_level': 'Safe',
            'confidence': 'High',
            'reasons': ['Verified legitimate organization domain present in trusted directory.'],
            'features': feat_dict,
            'display_features': display_features
        }
        
    # 2. Model Prediction
    artifact = get_model()
    if artifact and 'model' in artifact:
        model = artifact['model']
        X = pd.DataFrame([feat_dict], columns=FEATURE_COLUMNS)
        prob = model.predict_proba(X)[0][1] # Probability of phishing (class 1)
        risk_score = round(prob * 100, 1)
    else:
        # Heuristic fallback if model file is not present
        score = 15.0
        if feat_dict['is_ip']: score += 40
        if feat_dict['has_at_symbol']: score += 30
        if feat_dict['is_suspicious_tld']: score += 25
        if feat_dict['keyword_count'] > 0: score += (feat_dict['keyword_count'] * 15)
        if not feat_dict['is_https']: score += 10
        risk_score = min(99.0, max(1.0, score))
        
    if risk_score >= 65.0:
        risk_level = 'Dangerous'
        is_phishing = True
    elif risk_score >= 35.0:
        risk_level = 'Suspicious'
        is_phishing = False
    else:
        risk_level = 'Safe'
        is_phishing = False
        
    if not reasons:
        if is_phishing:
            reasons.append("Model detected suspicious structural patterns in the hostname and URL.")
        else:
            reasons.append("Standard domain structure and no high-risk markers detected.")

    return {
        'url': url,
        'host': host,
        'is_phishing': is_phishing,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'reasons': reasons,
        'features': feat_dict,
        'display_features': display_features
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    input_url = ""
    if request.method == 'POST':
        input_url = request.form.get('url', '').strip()
        if input_url:
            result = analyze_url(input_url)
    return render_template('index.html', result=result, input_url=input_url)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or request.form
    url = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'URL parameter is required.'}), 400
    result = analyze_url(url)
    return jsonify(result)

@app.route('/healthz', methods=['GET'])
@app.route('/health', methods=['GET'])
def health():
    artifact = get_model()
    return jsonify({
        'status': 'healthy',
        'model_loaded': artifact is not None,
        'metrics': artifact.get('metrics') if artifact else None
    })

if __name__ == '__main__':
    # Initial load of model
    get_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
