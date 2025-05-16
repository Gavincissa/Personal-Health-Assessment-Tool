### LINE 182 IS WHERE YOU CAN INPUT TEST DATA ###

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import tree

np.random.seed(42)

### Generate 9500 typical obese individuals
n_normal = 9500
heights_normal = np.clip(np.random.normal(170, 10, n_normal), 140, 200)
bmis_normal = np.random.uniform(30.1, 45.0, n_normal)
weights_normal = [bmi * ((h / 100) ** 2) for bmi, h in zip(bmis_normal, heights_normal)]

normal_df = pd.DataFrame({
    'Height': heights_normal,
    'Weight': weights_normal,
    'Glucose': np.clip(np.random.normal(100, 20, n_normal), 60, 200),
    'BloodPressure': np.clip(np.random.normal(125, 15, n_normal), 90, 200),
    'Cholesterol': np.clip(np.random.normal(210, 30, n_normal), 120, 350),
    'BoneDensity': np.clip(np.random.normal(1.2, 0.1, n_normal), 0.9, 1.6),
    'LeanMuscleMass': np.clip(np.random.normal(38, 5, n_normal), 20, 55)
})

### Add 500 healthy obese (fit-but-obese) individuals
n_healthy = 500
heights_healthy = np.clip(np.random.normal(175, 8, n_healthy), 160, 195)
bmis_healthy = np.random.uniform(30.1, 35.0, n_healthy)
weights_healthy = [bmi * ((h / 100) ** 2) for bmi, h in zip(bmis_healthy, heights_healthy)]

healthy_df = pd.DataFrame({
    'Height': heights_healthy,
    'Weight': weights_healthy,
    'Glucose': np.random.uniform(70, 95, n_healthy),
    'BloodPressure': np.random.uniform(100, 120, n_healthy),
    'Cholesterol': np.random.uniform(150, 200, n_healthy),
    'BoneDensity': np.random.uniform(1.3, 1.5, n_healthy),
    'LeanMuscleMass': np.random.uniform(50, 70, n_healthy)
})

### Add 200 high-risk lean obese individuals
n_high_risk = 200
heights_risk = np.clip(np.random.normal(170, 10, n_high_risk), 150, 185)
bmis_risk = np.random.uniform(30.5, 38.0, n_high_risk)
weights_risk = [bmi * ((h / 100) ** 2) for bmi, h in zip(bmis_risk, heights_risk)]

high_risk_df = pd.DataFrame({
    'Height': heights_risk,
    'Weight': weights_risk,
    'Glucose': np.random.uniform(130, 200, n_high_risk),
    'BloodPressure': np.random.uniform(140, 190, n_high_risk),
    'Cholesterol': np.random.uniform(250, 350, n_high_risk),
    'BoneDensity': np.random.uniform(1.3, 1.5, n_high_risk),
    'LeanMuscleMass': np.random.uniform(50, 65, n_high_risk)
})

### Combine all data
data = pd.concat([normal_df, healthy_df, high_risk_df], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)


def calculate_bmi(row):
    height_m = row['Height'] / 100
    return row['Weight'] / (height_m ** 2)

data['BMI'] = data.apply(calculate_bmi, axis=1)

def calculate_bodyfat_from_muscle(row):
    # Estimate base lean mass in kg
    lean_mass_kg = (row['LeanMuscleMass'] / 100) * row['Weight']

    # Apply bone density adjustment
    bone_factor = (row['BoneDensity'] - 1.2) * 10  # e.g., +1% per 0.1 above average

    adjusted_lean_mass = lean_mass_kg + (bone_factor * row['Weight'] / 100)

    body_fat_kg = row['Weight'] - adjusted_lean_mass
    body_fat_percent = (body_fat_kg / row['Weight']) * 100

    return round(body_fat_percent, 2)

data['BodyFatPercentage'] = data.apply(calculate_bodyfat_from_muscle, axis=1)


def compute_obesity_confidence(row):
    score = 0

    # Track critical markers
    critical_flags = 0
    if row['Glucose'] > 125:
        critical_flags += 1
    if row['Cholesterol'] > 240:
        critical_flags += 1
    if row['BodyFatPercentage'] >= 35:
        critical_flags += 1
    if row['BloodPressure'] > 130:
        critical_flags += 1

# Add bonus if multiple flags are triggered
    if critical_flags >= 3:
        score += 5

    # Clinical obesity indicators
    if row['Glucose'] >= 125:
        score += 15
    elif row['Glucose'] > 100:
        score += 10

    if row['Cholesterol'] > 240:
        score += 15
    elif row['Cholesterol'] > 200:
        score += 5

    if row['BodyFatPercentage'] >= 50:
        score += 62.5
    elif row['BodyFatPercentage'] >= 45:
        score += 50
    elif row['BodyFatPercentage'] >= 40:
        score += 42.5
    elif row['BodyFatPercentage'] >= 35:
        score += 35
    elif row['BodyFatPercentage'] >= 30:
        score += 27.5
    elif row['BodyFatPercentage'] >= 25:
        score += 15

    if row['BloodPressure'] > 130:
        score += 10

    # BMI is supportive, not primary
    if row['BMI'] > 40:
        score += 15
    elif row['BMI'] > 35:
        score += 10

    # Protective factors
    if row['BoneDensity'] > 1.3:
        score -= 10
    elif row['BoneDensity'] > 1.1:
        score -= 5


    return min(max(score, 0), 100)


data['ObesityConfidence'] = data.apply(compute_obesity_confidence, axis=1)

# Recreate features/labels
X = data.drop('ObesityConfidence', axis=1)
y = data['ObesityConfidence']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the regressor
reg = DecisionTreeRegressor(max_depth=15, random_state=42)
reg.fit(X_train, y_train)


# Predict and evaluate
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(
    reg,
    out_file=None,
    feature_names=X.columns,
    filled=True,
    rounded=True
)
graph = graphviz.Source(dot_data)
graph.render("obesity_tree")
graph.view()

# Input patient data here (You may change these values to test the model)
patient = pd.DataFrame([{
    'Glucose': 160, # mg/dL (Between 80-160 is valid)
    'BloodPressure': 180, # mmHg (Between 90-180 is valid)
    'Cholesterol': 300, # mg/dL (Between 120-300 is valid)
    'BoneDensity': 1.3, # g/cm^2 (Between 0.8-1.6 is valid)
    'LeanMuscleMass': 56, # % (Between 30-95 is valid)
    'Height': 190, # cm (Between 140-220 is valid)
    'Weight': 140, # kg (Between 40-160 is valid)
}])

# Checks if BMI is above 30
patient['BMI'] = patient.apply(calculate_bmi, axis=1)
if patient['BMI'].iloc[0] < 30:
    print("BMI is below 30, no need for further analysis.")
    exit()


else:
    patient['BodyFatPercentage'] = patient.apply(calculate_bodyfat_from_muscle, axis=1)

    print(f"Calculated Body Fat Percentage: {patient['BodyFatPercentage'].iloc[0]:.2f}%")
    print(f"Calculated BMI: {patient['BMI'].iloc[0]:.2f}\n")

    manual_score = compute_obesity_confidence(patient.iloc[0])
    print("Raw Score from logic:", manual_score)
    print(data.head())
    print("Min BMI:", data['BMI'].min())

    patient = patient[X.columns]

    high_score = reg.predict(patient)[0]
    print(f"Predicted Obesity Confidence Score: {high_score:.1f} / 100")
    print("Mean Squared Error:", mse) # MSE is a measure of how well the model predicts the target variable.




