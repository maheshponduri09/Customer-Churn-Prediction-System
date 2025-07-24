# 🔮 Customer Churn Prediction Project

A complete machine learning pipeline and web application for predicting customer churn, featuring advanced analytics, model interpretability, and batch/single prediction capabilities.

## 📦 Project Details

This project enables businesses to identify customers likely to churn, supporting proactive retention strategies. It includes data preprocessing, EDA, feature engineering, multiple ML models, and actionable insights.

### Main Components

- **Jupyter Notebook (`model.ipynb`)**: End-to-end ML workflow for churn prediction, including data analysis, feature engineering, model training, evaluation, interpretability (SHAP), and saving predictions with probability categories (`Low`, `Medium`, `High`).
- **Web App (`app.py`)**: Flask-based interface for model training, single/batch predictions, and CSV downloads.

## 🚀 Features

### Data Preprocessing & EDA
- Null value handling and imputation
- Label encoding for categorical variables
- Feature scaling (StandardScaler)
- Exploratory visualizations: churn distribution, age/balance/tenure analysis, correlation matrix

### Feature Engineering
- Age, balance, and tenure categorization
- Derived features: salary per product, balance per tenure, high-value indicator, product engagement

### Machine Learning Models
- Logistic Regression, Random Forest, XGBoost (in notebook)
- Model comparison with accuracy, precision, recall, F1, AUC, confusion matrix, ROC curves

### Model Interpretability
- SHAP summary plots and feature importance
- Top features for churn prediction

### Prediction Output
- Predictions saved to `churn_predictions.csv` with:
  - `CustomerId`
  - `Actual_Churn`
  - `Predicted_Churn`
  - `Churn_Probability`
  - `Churn_Probability_Category` (`Low`: <0.33, `Medium`: 0.33–0.66, `High`: >0.66)

### Web Interface
- Train model with uploaded CSV
- Predict churn for individual customers (JSON input)
- Batch prediction for CSV files, output includes probability and risk category (`Low`, `Medium`, `High`)
- Download batch prediction results
- Model status endpoint

## 🛠️ Installation & Setup

### Prerequisites

```bash
Python 3.8+
pip
```

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap flask werkzeug
```

### Project Structure

```
churn_prediction_project/
├── model.ipynb                # Jupyter notebook ML pipeline
├── app.py                     # Flask web application
├── templates/
│   └── index.html             # Web UI template
├── uploads/                   # File upload directory
├── churn_data.csv             # Example training dataset
├── churn_predictions.csv       # Notebook output predictions
├── batch_predictions.csv       # Web app batch prediction output
└── README.md                  # This file
```

## ⚡ Usage

### 1. Jupyter Notebook

Run `model.ipynb` for:
- Data loading, EDA, feature engineering
- Model training and evaluation
- SHAP interpretability
- Saving predictions with probability categories

### 2. Web Application

Start the Flask app:

```bash
python3 app.py
```

Open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).

#### Web Features

- **Train Model**: Upload CSV to train Random Forest model
- **Single Prediction**: Enter customer data for churn prediction (returns probability and risk category)
- **Batch Prediction**: Upload CSV for batch churn prediction (results include probability and risk category)
- **Download Results**: Export batch predictions as CSV
- **Model Status**: Check if model is trained and feature count

## 📑 Notes

- Churn probability is categorized as:
  - **Low**: < 0.33
  - **Medium**: 0.33–0.66
  - **High**: > 0.66
- Web app batch output uses slightly different thresholds (`Low`: <0.3, `Medium`: 0.3–0.7, `High`: >0.7).
- Ensure your input CSV matches required columns for training and prediction.

## 📬 Contact

For questions or feedback, please open an issue or contact the maintainer.