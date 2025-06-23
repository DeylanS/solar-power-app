import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Starting model training and saving process...")

# Load and prepare data set
df = pd.read_csv('Solar Power Plant Data.csv')
df['DateTime'] = pd.to_datetime(df['Date-Hour(NMT)'], format='%d.%m.%Y-%H:%M')
df['Hour'] = df['DateTime'].dt.hour
df['DayOfYear'] = df['DateTime'].dt.dayofyear

def categorize_power(power):
    if power == 0: return 'No Production'
    elif 1 <= power <= 2000: return 'Low'
    elif 2001 <= power <= 5000: return 'Medium'
    elif power > 5000: return 'High'
    return 'Undefined'
df['Power_Category'] = df['SystemProduction'].apply(categorize_power)

# Features used by Random Forest Classifier for best perfromance 
top_features = [
    'Radiation',
    'Hour',
    'AirTemperature',
    'DayOfYear',
    'Sunshine',
    'RelativeAirHumidity',
    'AirPressure'
]
X = df[top_features]
y = df['Power_Category']

# Scale the Data 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Final Model 
classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
classifier.fit(X_scaled, y)

# Save the model and the scaler
joblib.dump(classifier, 'solar_power_classifier.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler have been saved successfully!")
print("Saved files: solar_power_classifier.joblib, scaler.joblib")