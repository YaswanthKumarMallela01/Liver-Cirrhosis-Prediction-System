import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("Copy of HealthCareData.csv")
df.columns = df.columns.str.strip()  # Remove surrounding spaces

print("Available Columns (after stripping spaces):")
print(df.columns.tolist())

df.rename(columns={
    'Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)': 'Target'
}, inplace=True)

cat_map = {
    'positive': 1,
    'negative': 0,
    'yes': 1,
    'no': 0,
    'diffuse liver': 1,
    'normal': 0,
    'male': 1,
    'female': 0
}
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].map(cat_map).fillna(df[col])


df = df.apply(pd.to_numeric, errors='coerce')


df.dropna(thresh=len(df) * 0.3, axis=1, inplace=True)


df.fillna(df.median(numeric_only=True), inplace=True)


drop_cols = ['S.NO', 'Place(location where the patient lives)', 'Type of alcohol consumed', 'Blood pressure (mmhg)', 'TCH', 'TG', 'LDL', 'HDL', 'Hemoglobin  (g/dl)', 'PCV  (%)', 'RBC  (million cells/microliter)', 'MCV   (femtoliters/cell)', 'MCH  (picograms/cell)', 'MCHC  (grams/deciliter)', 'Total Count', 'Polymorphs  (%)', 'Lymphocytes  (%)', 'Monocytes   (%)', 'Eosinophils   (%)', 'Basophils  (%)', 'Indirect     (mg/dl)', 'Globulin  (g/dl)']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)


features = [
    "Age",
    "Gender",
    "Duration of alcohol consumption(years)",
    "Quantity of alcohol consumption (quarters/day)",
    "Hepatitis B infection",
    "Hepatitis C infection",
    "Diabetes Result",
    "Obesity",
    "Family history of cirrhosis/ hereditary",
    "Platelet Count  (lakhs/mm)",
    "Total Bilirubin    (mg/dl)",
    "Direct    (mg/dl)",
    "Total Protein     (g/dl)",
    "Albumin   (g/dl)",
    "A/G Ratio",
    "AL.Phosphatase      (U/L)",
    "SGOT/AST      (U/L)",
    "SGPT/ALT (U/L)",
    "USG Abdomen (diffuse liver or  not)"
]


missing = [f for f in features if f not in df.columns]
if missing:
    raise Exception(f"These required features are missing in the dataset: {missing}")


X = df[features]
y = df['Target']


normalizer = Normalizer(norm='l1')
X_normalized = normalizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


with open(f"rf_acc_{int(acc*100)}.pkl", "wb") as f:
    pickle.dump(model, f)
with open("normalizer.pkl", "wb") as f:
    pickle.dump(normalizer, f)

print("Model and normalizer saved.")
