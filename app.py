from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class ChurnPredictorWeb:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        
    def preprocess_single_record(self, data):
        """Preprocess a single customer record"""
        df = pd.DataFrame([data])
        
        # Encode gender if present
        if 'Gender' in df.columns:
            if 'Gender' in self.label_encoders:
                df['Gender'] = self.label_encoders['Gender'].transform(df['Gender'])
            else:
                le = LabelEncoder()
                df['Gender'] = le.fit_transform(df['Gender'])
                self.label_encoders['Gender'] = le
        
        # Remove CustomerID if present
        if 'CustomerId' in df.columns:
            df = df.drop('CustomerId', axis=1)
            
        return df
    
    def train_model(self, filepath):
        """Train the model with uploaded data"""
        try:
            # Load data
            df = pd.read_csv(filepath)

            # Select the specified columns
            required_columns = ['CustomerId', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']
            df = df[required_columns]
            
            # Preprocess data
            df_processed = df.copy()
            
            # Handle missing values
            df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))
            
            # Encode categorical variables
            if 'Gender' in df_processed.columns:
                le_gender = LabelEncoder()
                df_processed['Gender'] = le_gender.fit_transform(df_processed['Gender'])
                self.label_encoders['Gender'] = le_gender
            
            # Prepare features and target
            X = df_processed.drop(['CustomerId', 'Exited'], axis=1)
            y = df_processed['Exited']
            
            self.feature_names = X.columns.tolist()
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_scaled, y)
            
            self.is_trained = True
            
            # Calculate basic metrics
            accuracy = self.model.score(X_scaled, y)
            
            return {
                'success': True,
                'message': f'Model trained successfully! Training accuracy: {accuracy:.4f}',
                'data_shape': df.shape,
                'features': len(self.feature_names),
                'churn_rate': y.mean()
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error training model: {str(e)}'
            }
    
    def predict_churn(self, customer_data):
        """Predict churn for a single customer"""
        if not self.is_trained:
            return {'success': False, 'message': 'Model not trained yet'}
        
        try:
            # Preprocess the data
            df = self.preprocess_single_record(customer_data)
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    return {'success': False, 'message': f'Missing feature: {feature}'}
            
            # Scale the data
            X_scaled = self.scaler.transform(df[self.feature_names])
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0][1]
            
            # Determine risk category
            if probability < 0.3:
                risk_category = 'Low'
            elif probability < 0.7:
                risk_category = 'Medium'
            else:
                risk_category = 'High'
            
            return {
                'success': True,
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_category': risk_category,
                'message': f'Customer {"will likely churn" if prediction else "will likely stay"}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error making prediction: {str(e)}'
            }
    
    def predict_batch(self, filepath):
        """Predict churn for a batch of customers"""
        if not self.is_trained:
            return {'success': False, 'message': 'Model not trained yet'}
        
        try:
            # Load data
            df = pd.read_csv(filepath)

            # Select the specified columns
            required_columns = ['CustomerId', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
            df = df[required_columns]
            
            # Store original data for results
            original_df = df.copy()
            
            # Preprocess each row
            processed_rows = []
            for _, row in df.iterrows():
                processed_row = self.preprocess_single_record(row.to_dict())
                processed_rows.append(processed_row.iloc[0])
            
            processed_df = pd.DataFrame(processed_rows)
            
            # Make predictions
            X_scaled = self.scaler.transform(processed_df[self.feature_names])
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            
            # Add results to original dataframe
            results_df = original_df.copy()
            results_df['Predicted_Churn'] = predictions
            results_df['Churn_Probability'] = probabilities
            results_df['Risk_Category'] = pd.cut(probabilities, 
                                               bins=[0, 0.3, 0.7, 1.0], 
                                               labels=['Low', 'Medium', 'High'])
            
            # Save results
            output_path = 'batch_predictions.csv'
            results_df.to_csv(output_path, index=False)
            
            return {
                'success': True,
                'message': f'Batch prediction completed for {len(df)} customers',
                'output_file': output_path,
                'summary': {
                    'total_customers': len(df),
                    'predicted_churn': int(predictions.sum()),
                    'churn_rate': float(predictions.mean()),
                    'high_risk': int((probabilities > 0.7).sum()),
                    'medium_risk': int(((probabilities > 0.3) & (probabilities <= 0.7)).sum()),
                    'low_risk': int((probabilities <= 0.3).sum())
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error in batch prediction: {str(e)}'
            }

# Initialize the predictor
predictor = ChurnPredictorWeb()

@app.route('/')
def index():
    return render_template('index.html', is_trained=predictor.is_trained)

@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = predictor.train_model(filepath)
        return jsonify(result)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    result = predictor.predict_churn(data)
    return jsonify(result)

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = predictor.predict_batch(filepath)
        return jsonify(result)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(os.getcwd(), filename), as_attachment=True)

@app.route('/model_status')
def model_status():
    return jsonify({
        'is_trained': predictor.is_trained,
        'feature_count': len(predictor.feature_names) if predictor.is_trained else 0
    })

if __name__ == '__main__':
    app.run(debug=True)