from flask import Flask, render_template, request, jsonify
import joblib
import re
from urllib.parse import urlparse
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('phishing_model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None
    scaler = None

def extract_features(url):
    features = {}
    
    # Ensure URL has a scheme
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # URL length
    features['url_length'] = len(url)
    
    # Number of special characters
    features['special_char_count'] = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', url))
    
    # Number of digits
    features['digit_count'] = len(re.findall(r'\d', url))
    
    # Number of letters
    features['letter_count'] = len(re.findall(r'[a-zA-Z]', url))
    
    # Has https
    features['has_https'] = 1 if 'https' in url else 0
    
    # Has http
    features['has_http'] = 1 if 'http' in url and 'https' not in url else 0
    
    try:
        # Domain length
        parsed_url = urlparse(url)
        features['domain_length'] = len(parsed_url.netloc)
        
        # Number of subdomains
        features['subdomain_count'] = len(parsed_url.netloc.split('.')) - 1
    except Exception:
        # Fallback if URL parsing fails
        features['domain_length'] = len(url)
        features['subdomain_count'] = 0
    
    # Has @ symbol
    features['has_at'] = 1 if '@' in url else 0
    
    # Has IP address
    features['has_ip'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0
    
    return features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check_url', methods=['POST'])
def check_url():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded properly'}), 500
        
    try:
        url = request.form['url']
        
        # Extract features from the URL
        features = extract_features(url)
        features_df = pd.DataFrame([features])
        
        # Scale the features
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        result = {
            'is_phishing': bool(prediction),
            'probability': float(probability),
            'features': features
        }
        
        return jsonify(result)
    except Exception as e:
        print(f"Error analyzing URL: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Make sure we're in the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Current working directory:", os.getcwd())
    print("Starting Flask application...")
    app.run(host='127.0.0.1', port=5000, debug=True) 