# Phishing Website Detection System

This project implements a comprehensive phishing website detection system with advanced data preprocessing, visualization, and multiple machine learning models.

## Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA) with visualizations
- Feature importance analysis
- Multiple machine learning models:
  - Random Forest
  - XGBoost
  - LightGBM
  - CatBoost
  - Gradient Boosting
- Cross-validation
- Performance comparison
- Interactive visualizations

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Dataset

Place your dataset in the project directory with the name `dataset.csv`. The dataset should have the following format:
- Features: Various website characteristics
- Target: 'Result' column (0 for legitimate, 1 for phishing)

## Usage

1. Install the requirements:
```bash
pip install -r requirements.txt
```

2. Place your dataset in the project directory as `dataset.csv`

3. Run the main script:
```bash
python phishing_detection.py
```

## Output

The script will generate:
1. Various visualizations in the `plots` directory:
   - Target distribution
   - Correlation heatmap
   - Feature importance plot
   - Confusion matrices for each model
   - Model comparison plot
   - Interactive pair plot (HTML format)

2. Console output:
   - Data preprocessing information
   - Model performance metrics
   - Classification reports

## Improving Accuracy

The script implements several techniques to improve accuracy:
1. Feature scaling using StandardScaler
2. Multiple advanced models
3. Cross-validation
4. Feature importance analysis
5. Comprehensive data preprocessing

## Visualization Types

1. Distribution plots
2. Correlation heatmaps
3. Feature importance bar plots
4. Confusion matrices
5. Model comparison plots
6. Interactive pair plots

## Notes

- The script automatically handles missing values
- All visualizations are saved in the `plots` directory
- Cross-validation is performed with 5 folds
- The test set size is 20% of the total data 