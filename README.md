# Phishing-Website-Detection

A machine learning-based web application that detects potential phishing websites using URL features. The application provides a user-friendly interface where users can input URLs and get instant analysis results with interactive visualizations.

## Features

- Real-time URL analysis
- Modern web interface with interactive visualizations
- Detailed feature analysis
- Confidence score for predictions
- Responsive design
- Multiple data visualizations:
  - Feature distribution heatmap
  - Risk analysis radar chart
  - Feature importance pie chart
  - URL comparison scatter plot

## Project Structure

```
phishing-detector/
├── app.py              # Flask web application
├── train_model.py      # Model training script
├── phishing_detection.py # Feature extraction and model evaluation
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
│   └── index.html     # Main web interface
├── phishing_model.joblib # Trained model
├── scaler.joblib      # Feature scaler
└── README.md          # Project documentation
```

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. First, train the model:
```bash
python train_model.py
```

2. Start the web application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

4. Enter a URL in the input field and click "Check URL" to analyze it.

## Visualizations

The application provides several interactive visualizations to help understand the analysis:

1. **Feature Distribution Heatmap**: Shows the relative values of different URL features using color intensity.

2. **Risk Analysis Radar Chart**: Displays various risk factors on a radar chart, helping to identify which aspects of the URL are suspicious.

3. **Feature Importance Pie Chart**: Illustrates the distribution of key URL features, showing which characteristics are present in the analyzed URL.

4. **URL Comparison Scatter Plot**: Places the analyzed URL on a scatter plot alongside reference points for typical legitimate and phishing URLs, helping to visualize where the URL falls in relation to known patterns.

## How it Works

The application uses several features to detect phishing websites:
- URL length
- Special character count
- Digit count
- Letter count
- HTTPS/HTTP presence
- Domain length
- Subdomain count
- @ symbol presence
- IP address presence

The model is trained on the PhishTank dataset and uses a Random Forest classifier for predictions.

## Requirements

- Python 3.8+
- Flask
- scikit-learn
- pandas
- numpy
- joblib
- Chart.js (included via CDN)

## Note

This is a demonstration project and should not be used as the sole means of detecting phishing websites. Always use multiple security measures and stay vigilant when browsing the web. 

The dataset used here is : https://drive.google.com/file/d/1oFUge7ag89ZugAjiLDMUny17zULTp2_E/view?usp=sharing
