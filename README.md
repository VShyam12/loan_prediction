# Loan Eligibility Prediction Web Application

## Overview
This is a Flask-based web application that predicts loan eligibility using machine learning models. Users can input their personal and financial details, choose between XGBoost and Deep Neural Network models for prediction, and receive results along with risk factor analysis and improvement suggestions.

## Features
- **Dual Model Selection**: Choose between XGBoost and Deep Neural Network (DNN) models for prediction
- **Risk Factor Analysis**: Identifies key risk factors affecting loan eligibility
- **Improvement Suggestions**: Provides personalized recommendations to improve eligibility
- **Responsive UI**: Modern, user-friendly interface that works on all devices
- **Interactive Elements**: Dynamic form elements and visual feedback

## Project Structure
```
loan_flask/
├── app.py                  # Main Flask application
├── static/                 # Static assets
│   ├── css/
│   │   └── style.css       # Custom CSS styles
│   └── js/
│       └── script.js       # JavaScript for interactive elements
├── templates/              # HTML templates
│   ├── index.html          # Input form page
│   └── result.html         # Results display page
├── models/                 # Saved model files (created when running the app)
│   ├── xgboost_loan_model.pkl  # XGBoost model
│   ├── dnn_loan_model.pt       # PyTorch DNN model
│   ├── scaler.pkl             # Feature scaler
│   └── label_encoders.pkl     # Categorical encoders
└── README.md               # Project documentation
```

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. Clone the repository or download the files

2. Navigate to the project directory
   ```
   cd loan_flask
   ```

3. Create a virtual environment (optional but recommended)
   ```
   python -m venv venv
   ```

4. Activate the virtual environment
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

5. Install the required packages
   ```
   pip install -r requirements.txt
   ```
   If requirements.txt is not available, install the following packages:
   ```
   pip install flask pandas numpy scikit-learn xgboost torch joblib
   ```

### Running the Application

1. Start the Flask server
   ```
   python app.py
   ```

2. Open a web browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

## Usage

1. On the home page, fill in all the required fields with your personal and financial information
2. Select your preferred prediction model (XGBoost or DNN)
3. Click "Predict Eligibility"
4. View your results, including:
   - Loan approval status
   - Risk factors affecting your application
   - Suggestions for improving your eligibility
   - Summary of your input information

## Model Information

### XGBoost Model
The XGBoost model is a gradient boosting algorithm that excels at structured/tabular data. It's known for its performance and speed, making it a popular choice for financial predictions.

### Deep Neural Network (DNN) Model
The DNN model is a 1D convolutional neural network implemented in PyTorch. It can capture complex patterns in the data that traditional models might miss.

## Customization

- To modify the UI appearance, edit the CSS in `static/css/style.css`
- To change form fields or add new features, modify `templates/index.html` and update the corresponding routes in `app.py`
- To adjust the risk factor analysis, edit the `analyze_risk_factors` function in `app.py`

## License
This project is open source and available for personal and educational use.