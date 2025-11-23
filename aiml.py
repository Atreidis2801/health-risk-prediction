# Health Risk Prediction System
# A simple ML-based health assessment tool

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ==================== MODULE 1: DATA GENERATION ====================
class DataModule:
    """Handles data generation and preprocessing"""
    
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic patient data"""
        np.random.seed(42)
        
        data = {
            'age': np.random.randint(20, 80, n_samples),
            'bmi': np.random.uniform(18, 40, n_samples),
            'blood_pressure': np.random.randint(90, 180, n_samples),
            'glucose_level': np.random.randint(70, 200, n_samples),
            'cholesterol': np.random.randint(150, 300, n_samples),
            'heart_rate': np.random.randint(60, 100, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create risk label based on health indicators
        df['risk'] = (
            (df['age'] > 55).astype(int) +
            (df['bmi'] > 30).astype(int) +
            (df['blood_pressure'] > 140).astype(int) +
            (df['glucose_level'] > 140).astype(int) +
            (df['cholesterol'] > 240).astype(int)
        )
        df['risk'] = (df['risk'] >= 3).astype(int)  # 1 = High Risk, 0 = Low Risk
        
        return df
    
    def preprocess_data(self, df):
        """Split and scale the data"""
        X = df.drop('risk', axis=1)
        y = df['risk']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ==================== MODULE 2: MODEL TRAINING ====================
class ModelModule:
    """Handles model training and evaluation"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the risk prediction model"""
        print("Training model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Model training complete!")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if not self.is_trained:
            raise Exception("Model must be trained first!")
        
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print("\n=== Model Performance ===")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"\nConfusion Matrix:\n{cm}")
        
        return accuracy, precision, recall


# ==================== MODULE 3: PREDICTION & USER INTERFACE ====================
class PredictionModule:
    """Handles user input and predictions"""
    
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
    
    def get_user_input(self):
        """Get patient data from user"""
        print("\n" + "="*50)
        print("HEALTH RISK ASSESSMENT")
        print("="*50)
        
        try:
            age = int(input("Enter age (20-80): "))
            bmi = float(input("Enter BMI (18-40): "))
            bp = int(input("Enter blood pressure (90-180): "))
            glucose = int(input("Enter glucose level (70-200): "))
            cholesterol = int(input("Enter cholesterol (150-300): "))
            heart_rate = int(input("Enter heart rate (60-100): "))
            
            # Input validation
            if not (20 <= age <= 80):
                raise ValueError("Age out of range")
            if not (18 <= bmi <= 40):
                raise ValueError("BMI out of range")
            
            return {
                'age': age,
                'bmi': bmi,
                'blood_pressure': bp,
                'glucose_level': glucose,
                'cholesterol': cholesterol,
                'heart_rate': heart_rate
            }
        except ValueError as e:
            print(f"Invalid input: {e}")
            return None
    
    def predict_risk(self, patient_data):
        """Predict health risk for a patient"""
        if patient_data is None:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([patient_data])
        
        # Scale input
        df_scaled = self.scaler.transform(df)
        
        # Predict
        prediction = self.model.predict(df_scaled)[0]
        probability = self.model.predict_proba(df_scaled)[0]
        
        # Display results
        print("\n" + "="*50)
        print("RISK ASSESSMENT RESULTS")
        print("="*50)
        print(f"Patient Profile:")
        for key, value in patient_data.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print(f"\nRisk Level: {'HIGH RISK ⚠️' if prediction == 1 else 'LOW RISK ✓'}")
        print(f"Confidence: {probability[prediction]:.1%}")
        
        if prediction == 1:
            print("\nRecommendation: Consult a healthcare provider for detailed evaluation.")
        else:
            print("\nRecommendation: Maintain healthy lifestyle habits.")
        print("="*50)


# ==================== MAIN APPLICATION ====================
def main():
    """Main application workflow"""
    
    print("Initializing Health Risk Prediction System...")
    
    # Module 1: Data preparation
    data_module = DataModule()
    df = data_module.generate_sample_data(n_samples=1000)
    X_train, X_test, y_train, y_test, scaler = data_module.preprocess_data(df)
    
    # Module 2: Model training
    model_module = ModelModule()
    model_module.train(X_train, y_train)
    model_module.evaluate(X_test, y_test)
    
    # Module 3: User interaction
    prediction_module = PredictionModule(model_module.model, scaler)
    
    # Interactive prediction loop
    while True:
        patient_data = prediction_module.get_user_input()
        prediction_module.predict_risk(patient_data)
        
        continue_choice = input("\nAssess another patient? (yes/no): ").lower()
        if continue_choice != 'yes':
            print("\nThank you for using the Health Risk Prediction System!")
            break


if __name__ == "__main__":
    main()