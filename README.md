# Health Risk Prediction System

## Overview
A machine learning-based health assessment tool that predicts patient health risk levels based on key health indicators. The system uses Random Forest classification to categorize patients into high-risk or low-risk groups.

## Features
- **Data Processing**: Automated synthetic patient data generation and preprocessing
- **ML Model Training**: Random Forest classifier with 100 estimators
- **Risk Prediction**: Real-time health risk assessment based on 6 health parameters
- **Performance Metrics**: Accuracy, precision, recall, and confusion matrix evaluation
- **Interactive Interface**: Command-line user interface for patient data input
- **Input Validation**: Error handling for out-of-range values

## Technologies Used
- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms and preprocessing
- **RandomForestClassifier**: Core prediction model

## Installation & Setup

### Prerequisites
```bash
pip install pandas numpy scikit-learn
```

### Running the Project
1. Clone the repository:
```bash
git clone https://github.com/yourusername/health-risk-prediction.git
cd health-risk-prediction
```

2. Run the main script:
```bash
python health_risk_system.py
```

## Usage
1. The system will automatically:
   - Generate training data
   - Train the ML model
   - Display performance metrics

2. Enter patient health data when prompted:
   - Age (20-80 years)
   - BMI (18-40)
   - Blood Pressure (90-180 mmHg)
   - Glucose Level (70-200 mg/dL)
   - Cholesterol (150-300 mg/dL)
   - Heart Rate (60-100 bpm)

3. View risk assessment results and recommendations

## System Architecture

### Three Major Modules:
1. **DataModule**: Handles data generation, preprocessing, and scaling
2. **ModelModule**: Manages model training and performance evaluation
3. **PredictionModule**: Handles user interaction and risk predictions

## Testing
The system includes:
- Train/test split (80/20)
- Model evaluation with multiple metrics
- Input validation and error handling
- Automated testing on 200 test samples

## Sample Output
```
=== Model Performance ===
Accuracy: 87.50%
Precision: 85.20%
Recall: 83.40%

RISK ASSESSMENT RESULTS
Risk Level: HIGH RISK ⚠️
Confidence: 92.3%
Recommendation: Consult a healthcare provider
```

## Future Enhancements
- Web-based user interface
- Integration with real medical databases
- Support for more health parameters
- Multi-class risk categorization
- Export reports to PDF

## Author
Atreyi Barma Majumder
25BOE10040- AI/ML Project

## License
Educational Project - VITyarthi
