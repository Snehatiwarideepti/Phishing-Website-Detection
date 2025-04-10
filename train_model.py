import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import re
from urllib.parse import urlparse

def extract_features(url):
    features = {}
    
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
    features['has_http'] = 1 if 'http' in url else 0
    
    # Domain length
    parsed_url = urlparse(url)
    features['domain_length'] = len(parsed_url.netloc)
    
    # Number of subdomains
    features['subdomain_count'] = len(parsed_url.netloc.split('.')) - 1
    
    # Has @ symbol
    features['has_at'] = 1 if '@' in url else 0
    
    # Has IP address
    features['has_ip'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0
    
    return features

def train_model():
    print("Loading dataset...")
    df = pd.read_csv('phiusiil+phishing+url+dataset.csv')
    
    # Extract features from URLs
    print("Extracting features...")
    features_list = []
    for url in df['URL']:
        features = extract_features(url)
        features_list.append(features)
    
    # Convert features to DataFrame
    X = pd.DataFrame(features_list)
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save the model and scaler
    print("Saving model and scaler...")
    joblib.dump(model, 'phishing_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    # Print model performance
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"\nModel Performance:")
    print(f"Training Accuracy: {train_score:.4f}")
    print(f"Testing Accuracy: {test_score:.4f}")

if __name__ == "__main__":
    train_model() 