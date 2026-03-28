import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os

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

def train_models():
    print("Starting model training...")
    
    # Check if Loan_default.csv exists, if not create dummy data
    if os.path.exists("Loan_default.csv"):
        print("Loading real dataset...")
        df = pd.read_csv("Loan_default.csv")
    else:
        print("Creating dummy dataset for testing...")
        # Create dummy data with the same structure as the original dataset
        np.random.seed(42)
        n_samples = 1000
        
        # Generate dummy data
        data = {
            'LoanID': ['ID' + str(i).zfill(8) for i in range(n_samples)],
            'Age': np.random.randint(18, 80, n_samples),
            'Income': np.random.randint(20000, 200000, n_samples),
            'LoanAmount': np.random.randint(5000, 500000, n_samples),
            'CreditScore': np.random.randint(300, 850, n_samples),
            'MonthsEmployed': np.random.randint(0, 240, n_samples),
            'NumCreditLines': np.random.randint(0, 10, n_samples),
            'InterestRate': np.random.uniform(1, 25, n_samples),
            'LoanTerm': np.random.choice([12, 24, 36, 48, 60, 72, 84, 96, 120], n_samples),
            'DTIRatio': np.random.uniform(0.1, 0.9, n_samples),
            'Education': np.random.choice(['High School', 'Bachelor\'s', 'Master\'s', 'PhD'], n_samples),
            'EmploymentType': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], n_samples),
            'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
            'HasMortgage': np.random.choice(['Yes', 'No'], n_samples),
            'HasDependents': np.random.choice(['Yes', 'No'], n_samples),
            'LoanPurpose': np.random.choice(['Business', 'Education', 'Home', 'Other', 'Auto'], n_samples),
            'HasCoSigner': np.random.choice(['Yes', 'No'], n_samples),
            'Default': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 20% default rate
        }
        
        df = pd.DataFrame(data)
        df.to_csv("Loan_default.csv", index=False)
        print("Dummy dataset created and saved as Loan_default.csv")
    
    # Identify features (X) and target (y)
    X = df.drop(['LoanID', 'Default'], axis=1)
    y = df['Default']
    
    # Store label encoders
    label_encoders = {}
    
    # Apply Label Encoding to categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Apply SMOTE to balance the training data
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_bal_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved as scaler.pkl")
    
    # Save label encoders
    joblib.dump(label_encoders, 'label_encoders.pkl')
    print("Label encoders saved as label_encoders.pkl")
    
    # Train XGBoost model
    print("Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train_bal_scaled, y_train_bal)
    
    # Evaluate XGBoost model
    xgb_preds = xgb_model.predict(X_test_scaled)
    xgb_accuracy = accuracy_score(y_test, xgb_preds)
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
    
    # Save XGBoost model
    joblib.dump(xgb_model, 'xgboost_loan_model.pkl')
    print("XGBoost model saved as xgboost_loan_model.pkl")
    
    # Train DNN model
    print("Training DNN model...")
    
    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_bal_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_bal.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    # DataLoader for Training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize Model, Loss, Optimizer
    input_dim = X_train_tensor.shape[1]
    dnn_model = LoanCNN(input_features=input_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(dnn_model.parameters(), lr=0.001)
    
    # Training Loop
    epochs = 5  # Reduced for faster training
    dnn_model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = dnn_model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    # Evaluate DNN model
    dnn_model.eval()
    with torch.no_grad():
        predictions_logits = dnn_model(X_test_tensor)
        probabilities = torch.sigmoid(predictions_logits)
        predicted_classes = (probabilities >= 0.5).int()
        dnn_accuracy = accuracy_score(y_test.values, predicted_classes.numpy())
        print(f"DNN Accuracy: {dnn_accuracy:.4f}")
    
    # Save DNN model
    torch.save(dnn_model.state_dict(), 'dnn_loan_model.pt')
    print("DNN model saved as dnn_loan_model.pt")
    
    print("Model training complete!")

if __name__ == "__main__":
    train_models()