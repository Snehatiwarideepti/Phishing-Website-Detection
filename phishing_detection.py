 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from sklearn.metrics import roc_curve, auc

def main():
    try:
        # Set style for better visualizations
        plt.style.use('seaborn')
        sns.set_palette("husl")

        # Load the dataset
        print("Loading dataset...")
        df = pd.read_csv('phiusiil+phishing+url+dataset.csv')

        # Display basic information about the dataset
        print("\nDataset Info:")
        print(df.info())

        # Display first few rows
        print("\nFirst few rows of the dataset:")
        print(df.head())

        # Check for missing values
        print("\nMissing values:")
        print(df.isnull().sum())

        # Basic statistics
        print("\nBasic statistics:")
        print(df.describe())

        # Prepare the data
        X = df.drop('label', axis=1)
        y = df['label']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest model
        print("\nTraining Random Forest model...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)

        # Train XGBoost model
        print("\nTraining XGBoost model...")
        xgb_model = xgb.XGBClassifier(random_state=42)
        xgb_model.fit(X_train_scaled, y_train)

        # Make predictions
        rf_pred = rf_model.predict(X_test_scaled)
        xgb_pred = xgb_model.predict(X_test_scaled)

        # Print classification reports
        print("\nRandom Forest Classification Report:")
        print(classification_report(y_test, rf_pred))

        print("\nXGBoost Classification Report:")
        print(classification_report(y_test, xgb_pred))

        # Create visualizations
        create_visualizations(df)
        update_visualizations(rf_model, xgb_model, X_test_scaled, y_test, df, X)

        print("\nVisualizations have been saved as 'visualizations.png' and 'model_results.png'")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def create_visualizations(df):
    # Create a figure with multiple subplots
    plt.figure(figsize=(20, 15))
    
    # 1. Distribution of target variable
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x='label')
    plt.title('Distribution of Phishing vs Legitimate Websites')
    plt.xlabel('Label (0: Legitimate, 1: Phishing)')
    plt.ylabel('Count')
    
    # 2. Correlation heatmap
    plt.subplot(2, 2, 2)
    correlation = df.corr()
    sns.heatmap(correlation, cmap='coolwarm', annot=False)
    plt.title('Feature Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig('visualizations.png')
    plt.close()

def update_visualizations(rf_model, xgb_model, X_test_scaled, y_test, df, X):
    plt.figure(figsize=(20, 15))
    
    # 1. Distribution of target variable
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x='label')
    plt.title('Distribution of Phishing vs Legitimate Websites')
    plt.xlabel('Label (0: Legitimate, 1: Phishing)')
    plt.ylabel('Count')
    
    # 2. Correlation heatmap
    plt.subplot(2, 2, 2)
    correlation = df.corr()
    sns.heatmap(correlation, cmap='coolwarm', annot=False)
    plt.title('Feature Correlation Heatmap')
    
    # 3. Feature importance plot
    plt.subplot(2, 2, 3)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features')
    
    # 4. ROC curves
    plt.subplot(2, 2, 4)
    
    # Random Forest ROC
    rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
    rf_auc = auc(rf_fpr, rf_tpr)
    
    # XGBoost ROC
    xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)
    xgb_auc = auc(xgb_fpr, xgb_tpr)
    
    plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
    plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_results.png')
    plt.close()

if __name__ == "__main__":
    main()