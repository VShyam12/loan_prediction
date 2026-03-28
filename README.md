💰 Loan Eligibility Prediction Web Application

A Flask-based web application that predicts whether a loan applicant is likely to default or not using Machine Learning and Deep Learning models.

🚀 Project Overview

This project aims to assist financial institutions in automating loan approval decisions by analyzing applicant data and predicting the likelihood of loan default.

The system uses trained ML/DL models and provides predictions through a user-friendly web interface.

🧠 Technologies Used
Backend: Flask (Python)
Frontend: HTML, CSS, JavaScript
Machine Learning: Scikit-learn, XGBoost
Deep Learning: PyTorch
Data Processing: Pandas, NumPy

⚙️ Features
🔍 Predict loan eligibility instantly
📊 Uses both ML and DL models for prediction
🖥️ Simple and intuitive web interface
⚡ Fast and efficient predictions
📁 Modular project structure

📂 Project Structure
```
loan_prediction/
├── app.py                      # Main Flask application
├── train_models.py             # Script to train ML/DL models
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── templates/                  # HTML files
│   ├── index.html
│   └── result.html
├── static/                     # CSS & JS
│   ├── css/
│   └── js/
└── Loan_Prediction.ipynb       # Model development notebook
```
🧪 Machine Learning Models Used
XGBoost Classifier
Deep Neural Network (PyTorch)

These models are trained on historical loan data to classify applicants as:
✅ Eligible (Low Risk)
❌ Not Eligible (High Risk / Default)

▶️ How to Run the Project Locally
1️⃣ Clone the Repository
git clone https://github.com/VShyam12/loan_prediction.git
cd loan_prediction
2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Train Models (Important)

Since model files are not included in the repository:
python train_models.py
5️⃣ Run the Flask App
python app.py
6️⃣ Open in Browser
http://127.0.0.1:5000/
⚠️ Important Notes
🚫 Model files (.pkl, .pt) are excluded for security and size reasons
🚫 Dataset (.csv) is not included in the repository
✔️ Generate models locally using train_models.py

🌐 Future Improvements
Deploy the app on cloud (Render / AWS)
Add user authentication
Improve model accuracy with more data
Add API endpoints for integration
🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.

📜 License

This project is for educational purposes.

👨‍💻 Author

Shyam
GitHub: https://github.com/VShyam12
