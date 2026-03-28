from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import os

app = Flask(__name__)

# Define feature names and their types
feature_names = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
    'Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
    'HasDependents', 'LoanPurpose', 'HasCoSigner'
]

numerical_features = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
]

categorical_features_options = {
    'Education': ['High School', 'Bachelor\'s', 'Master\'s', 'PhD'],
    'EmploymentType': ['Full-time', 'Part-time', 'Self-employed', 'Unemployed'],
    'MaritalStatus': ['Single', 'Married', 'Divorced'],
    'HasMortgage': ['No', 'Yes'],
    'HasDependents': ['No', 'Yes'],
    'LoanPurpose': ['Business', 'Education', 'Home', 'Other', 'Auto'],
    'HasCoSigner': ['No', 'Yes']
}

# Define the CNN model for loan prediction
class LoanCNN(nn.Module):
    def __init__(self, input_features):
        super(LoanCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self._to_linear = None
        self._calculate_flat_features(input_features)
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def _calculate_flat_features(self, input_features):
        x = torch.randn(1, 1, input_features)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        self._to_linear = x.numel()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)

# Load models and preprocessors
def load_models():
    models = {}
    preprocessors = {}
    
    # Check if models exist, if not, create dummy models for testing
    if os.path.exists('xgboost_loan_model.pkl'):
        try:
            models['xgboost'] = joblib.load('xgboost_loan_model.pkl')
        except:
            # Create a dummy XGBoost model if loading fails
            models['xgboost'] = xgb.XGBClassifier()
    else:
        # Create a dummy XGBoost model
        models['xgboost'] = xgb.XGBClassifier()
    
    # Load or create DNN model
    if os.path.exists('dnn_loan_model.pt'):
        input_dim = len(feature_names)
        models['dnn'] = LoanCNN(input_features=input_dim)
        models['dnn'].load_state_dict(torch.load('dnn_loan_model.pt'))
        models['dnn'].eval()
    else:
        # Create a dummy DNN model
        input_dim = len(feature_names)
        models['dnn'] = LoanCNN(input_features=input_dim)
        models['dnn'].eval()
    
    # Load or create preprocessors
    if os.path.exists('scaler.pkl'):
        preprocessors['scaler'] = joblib.load('scaler.pkl')
    else:
        preprocessors['scaler'] = StandardScaler()
    
    if os.path.exists('label_encoders.pkl'):
        preprocessors['label_encoders'] = joblib.load('label_encoders.pkl')
    else:
        # Create dummy label encoders
        label_encoders = {}
        for col in categorical_features_options.keys():
            le = LabelEncoder()
            le.fit(categorical_features_options[col])
            label_encoders[col] = le
        preprocessors['label_encoders'] = label_encoders
    
    return models, preprocessors

# Load models at startup
models, preprocessors = load_models()

# Calculate risk score based on all features
def calculate_risk_score(input_data):
    risk_score = 0
    
    # Loan-to-Income Ratio Impact (0-40 points) - PRIMARY factor (increased importance)
    loan_to_income = input_data['LoanAmount'] / input_data['Income']
    if loan_to_income > 3:
        risk_score += 40  # Extremely high ratio (increased weight)
    elif loan_to_income > 2:
        risk_score += 28  # High ratio (slightly increased)
    elif loan_to_income > 1.5:
        risk_score += 22  # Moderate-high ratio (slightly increased)
    elif loan_to_income > 1:
        risk_score += 18  # Moderate ratio
    elif loan_to_income > 0.5:
        risk_score += 10   # Low-moderate ratio
    elif loan_to_income > 0.3:
        risk_score += 4   # Low ratio
    # Very low ratio (< 0.3) adds 0
    
    # Credit Score Impact (0-25 points) - Reduced weight
    if input_data['CreditScore'] < 580:
        risk_score += 25  # Very poor credit
    elif input_data['CreditScore'] < 620:
        risk_score += 18  # Poor credit
    elif input_data['CreditScore'] < 670:
        risk_score += 12  # Fair credit
    elif input_data['CreditScore'] < 740:
        risk_score += 6   # Good credit
    elif input_data['CreditScore'] < 800:
        risk_score += 2   # Very good credit
    # Excellent credit adds 0
    
    # DTI Ratio Impact (0-20 points) - Reduced weight
    if input_data['DTIRatio'] > 0.50:
        risk_score += 20  # Very high DTI
    elif input_data['DTIRatio'] > 0.43:
        risk_score += 15  # High DTI
    elif input_data['DTIRatio'] > 0.36:
        risk_score += 10  # Moderate-high DTI
    elif input_data['DTIRatio'] > 0.28:
        risk_score += 5   # Moderate DTI
    elif input_data['DTIRatio'] > 0.20:
        risk_score += 2   # Low-moderate DTI
    # Very low DTI adds 0
    
    # Employment Status Impact (0-12 points)
    if input_data['EmploymentType'] == 'Unemployed':
        risk_score += 12  # Unemployed
    elif input_data['EmploymentType'] == 'Part-time':
        risk_score += 8   # Part-time
    elif input_data['EmploymentType'] == 'Self-employed':
        risk_score += 5   # Self-employed
    # Full-time adds 0
    
    # Employment Duration Impact (0-10 points) - Reduced weight
    if input_data['MonthsEmployed'] < 6:
        risk_score += 10  # Less than 6 months
    elif input_data['MonthsEmployed'] < 12:
        risk_score += 6   # 6-12 months
    elif input_data['MonthsEmployed'] < 24:
        risk_score += 3   # 1-2 years
    # More than 2 years adds 0
    
    # Interest Rate Impact (0-10 points) - Reduced weight
    if input_data['InterestRate'] > 15:
        risk_score += 10  # Very high interest rate
    elif input_data['InterestRate'] > 12:
        risk_score += 7   # High interest rate
    elif input_data['InterestRate'] > 9:
        risk_score += 4   # Moderate-high interest rate
    elif input_data['InterestRate'] > 6:
        risk_score += 2   # Moderate interest rate
    # Low interest rate adds 0
    
    # Loan Term Impact (0-8 points) - Reduced weight
    if input_data['LoanTerm'] > 60:
        risk_score += 8   # Very long term
    elif input_data['LoanTerm'] > 48:
        risk_score += 5   # Long term
    elif input_data['LoanTerm'] > 36:
        risk_score += 2   # Moderate term
    # Short term adds 0
    
    return risk_score

# Risk factors analysis
def analyze_risk_factors(input_data):
    risk_factors = []
    improvements = []
    
    # Credit Score Analysis
    if input_data['CreditScore'] < 580:
        risk_factors.append("Very Low Credit Score (Critical)")
        improvements.append("Work on improving your credit score by paying bills on time and reducing debt. This is a critical factor for loan approval.")
    elif input_data['CreditScore'] < 670:
        # Use less alarming wording for the 'fair' score bucket
        risk_factors.append("Fair Credit Score (Some Risk)")
        improvements.append("Continue building your credit history and avoid new debt. Most lenders prefer scores above 670.")
    elif input_data['CreditScore'] < 740:
        risk_factors.append("Good Credit Score (Moderate Risk)")
        improvements.append("Your credit score is good, but improving it further could qualify you for better rates.")
    
    # DTI Ratio Analysis
    if input_data['DTIRatio'] > 0.43:
        risk_factors.append("Very High Debt-to-Income Ratio (Critical)")
        improvements.append("Your debt payments are too high relative to your income. Reduce your existing debt or increase your income significantly.")
    elif input_data['DTIRatio'] > 0.36:
        risk_factors.append("High Debt-to-Income Ratio (High Risk)")
        improvements.append("Your DTI ratio exceeds recommended levels. Work on reducing debt or increasing income.")
    elif input_data['DTIRatio'] > 0.28:
        risk_factors.append("Moderate Debt-to-Income Ratio (Moderate Risk)")
        improvements.append("Consider reducing some debt to improve your DTI ratio below 28% for better loan terms.")
    
    # Employment Analysis
    if input_data['EmploymentType'] == 'Unemployed':
        risk_factors.append("Unemployment (Critical)")
        improvements.append("Secure stable employment to improve loan eligibility. Lenders require reliable income sources.")
    elif input_data['EmploymentType'] == 'Part-time':
        risk_factors.append("Part-time Employment (High Risk)")
        improvements.append("Consider seeking full-time employment to demonstrate stable income.")
    elif input_data['EmploymentType'] == 'Self-employed':
        risk_factors.append("Self-employment (Moderate Risk)")
        improvements.append("Provide additional documentation of stable income history and business performance.")
    
    if input_data['MonthsEmployed'] < 6:
        risk_factors.append("Very Short Employment History (Critical)")
        improvements.append("Most lenders require at least 6 months of employment history. Wait until you have a longer history.")
    elif input_data['MonthsEmployed'] < 12:
        risk_factors.append("Short Employment History (High Risk)")
        improvements.append("A longer employment history demonstrates stability to lenders. Try to maintain your current position.")
    elif input_data['MonthsEmployed'] < 24:
        risk_factors.append("Moderate Employment History (Moderate Risk)")
        improvements.append("While acceptable, 2+ years at the same employer is preferred for optimal loan terms.")
    
    # Loan Amount to Income Ratio
    loan_to_income = input_data['LoanAmount'] / input_data['Income']
    if loan_to_income > 3:
        risk_factors.append("Extremely High Loan-to-Income Ratio (Critical)")
        improvements.append("Your loan amount is more than 3 times your annual income. This is a critical factor for loan denial. Consider a much smaller loan or significantly higher income.")
    elif loan_to_income > 2:
        risk_factors.append("High Loan-to-Income Ratio (High Risk)")
        improvements.append("Your loan amount exceeds twice your annual income. Consider a smaller loan amount or work on increasing your income.")
    elif loan_to_income > 1.5:
        risk_factors.append("Moderate-High Loan-to-Income Ratio (Moderate Risk)")
        improvements.append("Your loan amount is relatively high compared to your income. Consider if a smaller loan would meet your needs.")
    
    # Interest Rate Analysis
    if input_data['InterestRate'] > 12:
        risk_factors.append("Very High Interest Rate (Critical)")
        improvements.append("Your interest rate is extremely high. Shop around for better rates or improve your credit profile significantly.")
    elif input_data['InterestRate'] > 9:
        risk_factors.append("High Interest Rate (High Risk)")
        improvements.append("Your interest rate is above average. Improving your credit score could help secure better rates.")
    elif input_data['InterestRate'] > 6:
        risk_factors.append("Moderate Interest Rate (Moderate Risk)")
        improvements.append("Your interest rate is reasonable but could be improved with a better credit profile.")
    
    # Loan Term Analysis
    if input_data['LoanTerm'] > 60:
        risk_factors.append("Very Long Loan Term (High Risk)")
        improvements.append("Long loan terms increase total interest paid and risk of being underwater. Consider a shorter term if possible.")
    elif input_data['LoanTerm'] > 36:
        risk_factors.append("Moderate-Long Loan Term (Moderate Risk)")
        improvements.append("Consider if a shorter loan term would be more economical in the long run.")
    
    # Co-signer Analysis
    if input_data['HasCoSigner'] == 'No':
        if input_data['CreditScore'] < 650 or input_data['DTIRatio'] > 0.36 or loan_to_income > 2:
            improvements.append("Consider getting a co-signer with good credit to improve approval chances.")
    
    return risk_factors, improvements

# Preprocess input data
def preprocess_data(input_data, model_type):
    # Create DataFrame from input data
    input_values_ordered = [input_data[feat] for feat in feature_names]
    input_df = pd.DataFrame([input_values_ordered], columns=feature_names)
    
    # Apply Label Encoding to categorical features
    label_encoders = preprocessors['label_encoders']
    for col in categorical_features_options.keys():
        if col in input_df.columns:
            le = label_encoders[col]
            input_df[col] = le.transform([input_df[col].iloc[0]])[0]
    
    # Scale all features (both numerical and categorical)
    # This matches how the scaler was trained in train_models.py
    scaler = preprocessors['scaler']
    scaled_data = scaler.transform(input_df.values)
    input_df = pd.DataFrame(scaled_data, columns=input_df.columns)
    
    if model_type == 'dnn':
        # Convert to PyTorch tensor for DNN model
        return torch.tensor(input_df.values, dtype=torch.float32)
    else:
        # Return DataFrame for XGBoost model
        return input_df

# Make prediction
def predict(input_data, model_type):
    # Calculate risk score based on all features
    risk_score = calculate_risk_score(input_data)
    
    # Check for automatic rejection conditions
    loan_to_income = input_data['LoanAmount'] / input_data['Income']
    
    # Strict automatic rejection conditions - only for extreme cases
    if loan_to_income > 4:
        # Return (approved=False, final_probability, model_confidence)
        return False, 0.05, 0.0  # Automatic rejection for extremely high loan-to-income ratio (more than 4x income)

    if input_data['CreditScore'] < 550 and input_data['DTIRatio'] > 0.5:
        return False, 0.08, 0.0  # Automatic rejection for very poor credit + very high DTI

    if input_data['EmploymentType'] == 'Unemployed' and loan_to_income > 0.5:
        return False, 0.05, 0.0  # Automatic rejection for unemployed with significant loan amount
    
    # Process data through the model
    processed_data = preprocess_data(input_data, model_type)
    
    if model_type == 'dnn':
        model = models['dnn']
        with torch.no_grad():
            prediction_logits = model(processed_data)
            base_probability = torch.sigmoid(prediction_logits).item()
    else:  # xgboost
        model = models['xgboost']
        try:
            # Try to get prediction probability from XGBoost model
            proba = model.predict_proba(processed_data)
            base_probability = proba[:, 1][0] if len(proba.shape) > 1 else proba[0]
        except:
            # Fallback: use more realistic probability based on risk factors
            base_probability = max(0.3, 1 - (risk_score / 100))
    
    # Combine model confidence and risk score into a final eligibility probability
    # Normalize risk_score to [0,1]
    norm_risk = min(max(risk_score, 0), 100) / 100.0

    # weight: how strongly risk reduces model confidence (0.0 - 1.0)
    risk_weight = 0.65

    # final probability is model base probability reduced by risk factor
    final_probability = base_probability * (1.0 - risk_weight * norm_risk)

    # Apply loan_to_income multipliers as additional business rule (stronger weight)
    if loan_to_income > 3:
        final_probability *= 0.35
    elif loan_to_income > 2:
        final_probability *= 0.65
    elif loan_to_income > 1.5:
        final_probability *= 0.88

    # Ensure final probability in [0,1]
    final_probability = float(min(max(final_probability, 0.0), 1.0))

    # Make final prediction (approved = True) using 50% threshold
    approved = True if final_probability >= 0.5 else False
    # Debug logging: print intermediate values so results are auditable in the server logs
    try:
        # Use print so values appear in the console where app.py is run
        print(f"[PREDICT DEBUG] model_type={model_type} base_probability={float(base_probability):.4f} risk_score={risk_score} norm_risk={norm_risk:.4f} loan_to_income={loan_to_income:.4f} final_probability={final_probability:.4f} approved={approved}")
    except Exception:
        pass

    return approved, final_probability, float(base_probability)

@app.route('/')
def index():
    return render_template('index.html', 
                           categorical_features=categorical_features_options)

@app.route('/predict', methods=['POST'])
def predict_loan():
    if request.method == 'POST':
        # Get form data
        input_data = {}
        
        # Get numerical features
        for feature in numerical_features:
            input_data[feature] = float(request.form.get(feature, 0))
        
        # Get categorical features
        for feature in categorical_features_options.keys():
            input_data[feature] = request.form.get(feature, '')
        
        # Get selected model
        model_type = request.form.get('model_type', 'xgboost')
        
        # Calculate risk score for display
        risk_score = calculate_risk_score(input_data)

        # Make prediction
        approved, probability, model_confidence = predict(input_data, model_type)

        # Analyze risk factors
        risk_factors, improvements = analyze_risk_factors(input_data)

        return render_template('result.html',
                              approved=approved,
                              probability=probability,
                              model_confidence=model_confidence,
                              risk_factors=risk_factors,
                              improvements=improvements,
                              input_data=input_data,
                              model_type=model_type,
                              risk_score=risk_score)

if __name__ == '__main__':
    app.run(debug=True)