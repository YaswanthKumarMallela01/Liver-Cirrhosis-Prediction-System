import pickle
import numpy as np
from flask import Flask, render_template, request


model = pickle.load(open("rf_acc_100.pkl", "rb"))
normalizer = pickle.load(open("normalizer.pkl", "rb"))


input_features = [
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

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form.get(f)) for f in input_features]
        input_array = np.array(input_data).reshape(1, -1)
        input_normalized = normalizer.transform(input_array)

        prediction = model.predict(input_normalized)[0]
        probability = model.predict_proba(input_normalized)[0][int(prediction)]

        label = "🟢 Negative for Liver Cirrhosis" if prediction == 0 else "🔴 Positive for Liver Cirrhosis"
        tip = (
            "Your liver health indicators seem normal. Maintain a healthy diet and regular checkups." if prediction == 0
            else "Your inputs show risk markers. We recommend you consult a hepatologist for further diagnosis."
        )

        return render_template(
            'index.html',
            prediction_text=label,
            confidence=f"{probability * 100:.2f}%",
            health_tip=tip,
            is_error=False
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Error: {e}", is_error=True)


if __name__ == "__main__":
    app.run(debug=True)
