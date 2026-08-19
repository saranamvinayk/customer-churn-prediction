import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the Data
print("Loading data...")
df = pd.read_csv('data/telco.csv')

# 2. Preprocessing & Cleaning (The 'Gotcha' Fix)
print("Cleaning data...")
# Drop customerID as it has no predictive value
df = df.drop('customerID', axis=1)

# Fix the TotalCharges column: coerce errors to NaN, then fill with 0
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Map the target variable 'Churn' from Yes/No to 1/0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 3. Feature Engineering (One-Hot Encoding)
print("Encoding categorical features...")
# Convert all remaining text columns (like InternetService, Contract) into 1s and 0s
X = pd.get_dummies(df.drop('Churn', axis=1), drop_first=True)
y = df['Churn']

# 4. Train/Test Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Scaling Numerical Features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Model Training
print("Training Random Forest model...")
# class_weight='balanced' is crucial because fewer people churn than stay. 
# This forces the model to pay extra attention to the minority class.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)

# 7. Evaluation & Metrics
print("\n--- Model Evaluation ---")
y_pred = rf_model.predict(X_test_scaled)

# Print the Classification Report
print(classification_report(y_test, y_pred))

# Plot the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Retained', 'Churned'], 
            yticklabels=['Retained', 'Churned'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Truth')
plt.xlabel('Model Prediction')
plt.show()
