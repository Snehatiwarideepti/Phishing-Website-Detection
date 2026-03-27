import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Check for missing values
    print("\nMissing values in dataset:")
    print(df.isnull().sum())
    
    # Handle missing values if any
    df = df.fillna(df.mean())
    
    # Separate features and target
    X = df.drop('Result', axis=1)  # Assuming 'Result' is the target column
    y = df['Result']
    
    return X, y, df

def perform_eda(df):
    """
    Perform Exploratory Data Analysis
    """
    # Create a directory for saving plots
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # 1. Distribution of target variable
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Result')
    plt.title('Distribution of Phishing vs Legitimate Websites')
    plt.savefig('plots/target_distribution.png')
    plt.close()
    
    # 2. Correlation heatmap
    plt.figure(figsize=(15, 12))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()
    
    # 3. Feature importance visualization using Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(df.drop('Result', axis=1), df['Result'])
    
    feature_importance = pd.DataFrame({
        'feature': df.drop('Result', axis=1).columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()
    
    # 4. Interactive pair plot for top features
    top_features = feature_importance['feature'].head(5).tolist()
    top_features.append('Result')
    pair_plot = px.scatter_matrix(df[top_features], color='Result')
    pair_plot.write_html('plots/pair_plot.html')

def train_and_evaluate_models(X, y):
    """
    Train and evaluate multiple models
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=False),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        results[name] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'plots/confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close()
    
    return results

def plot_model_comparison(results):
    """
    Create comparison plots for different models
    """
    # Prepare data for plotting
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    cv_means = [results[model]['cv_mean'] for model in models]
    cv_stds = [results[model]['cv_std'] for model in models]
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Test Accuracy')
    plt.bar(x + width/2, cv_means, width, label='CV Mean Accuracy')
    plt.errorbar(x + width/2, cv_means, yerr=cv_stds, fmt='none', color='black', capsize=5)
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png')
    plt.close()

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, df = load_and_preprocess_data('dataset.csv')  # Replace with your dataset path
    
    # Perform EDA
    print("\nPerforming Exploratory Data Analysis...")
    perform_eda(df)
    
    # Train and evaluate models
    print("\nTraining and evaluating models...")
    results = train_and_evaluate_models(X, y)
    
    # Plot model comparison
    print("\nCreating model comparison plots...")
    plot_model_comparison(results)
    
    # Print final results
    print("\nFinal Results:")
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"Test Accuracy: {result['accuracy']:.4f}")
        print(f"Cross-validation Mean Accuracy: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})")
        print("\nClassification Report:")
        print(result['classification_report'])

if __name__ == "__main__":
    main() 