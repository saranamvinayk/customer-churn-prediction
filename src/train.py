import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import mlflow
import mlflow.sklearn
import warnings

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

# 1. Load the Data
print("Loading data...")
df = pd.read_csv('data/telco.csv')

# 2. Prevent Data Leakage and Remove Noise
print("Cleaning data and dropping leakage columns...")
columns_to_drop = [
    'Customer ID',       # No predictive value
    'Country', 'State', 'City', 'Zip Code', 'Latitude', 'Longitude', 'Population', # Geographic noise
    'Quarter',           # Temporal noise
    'Customer Status',   # Data Leakage (Directly reveals if they churned)
    'Churn Score',       # Data Leakage (An existing model's output)
    'Churn Category',    # Data Leakage (Why they churned)
    'Churn Reason'       # Data Leakage (Why they churned)
]

# Drop the columns (errors='ignore' prevents crashes if a column name has a slight typo)
df = df.drop(columns=columns_to_drop, errors='ignore')

# 3. Handle Target Variable & Data Types
# Map the target 'Churn Label' to 1 (Yes) and 0 (No)
df['Churn Label'] = df['Churn Label'].map({'Yes': 1, 'No': 0})

# Drop any rows where the target might be missing
df = df.dropna(subset=['Churn Label'])

# Fix 'Total Charges' (often loaded as strings due to blank spaces)
if 'Total Charges' in df.columns:
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce').fillna(0)

# 4. Feature Engineering (One-Hot Encoding)
print("Encoding categorical features...")
X = pd.get_dummies(df.drop('Churn Label', axis=1), drop_first=True)
y = df['Churn Label']

# 5. Train/Test Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Scaling
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.joblib')

# 7. MLflow Tracking
mlflow.set_experiment("Telco_Churn_Prediction")

n_estimators = 150
max_depth = 8

print("Starting MLflow run...")
with mlflow.start_run(run_name="RandomForest_IBM_Dataset"):
    
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("class_weight", "balanced")
    
    # Train the model
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42, 
        class_weight='balanced'
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = rf_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    
    # Log the model to MLflow
    mlflow.sklearn.log_model(rf_model, "random_forest_model")
    
    print(f"Run completed successfully! F1 Score: {f1:.4f}")
