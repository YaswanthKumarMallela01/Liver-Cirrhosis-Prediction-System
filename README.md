# Liver Cirrhosis Prediction Web Application

**Deployed Live App:** https://yashs-liver-cirrhosis-prediction-system-uvwq.onrender.com

## Project Description
This project implements a web application using Flask to predict the likelihood of liver cirrhosis based on various health indicators. It leverages a machine learning model (RandomForestClassifier) trained on a comprehensive dataset of medical parameters. The application provides a user-friendly interface for inputting patient data and receiving real-time predictions, along with a confidence score and a health tip.

## Key Features
*   **Web Interface**: A clean and interactive web form built with Flask and HTML/CSS for easy data input.
*   **Machine Learning Model**: Utilizes a pre-trained `RandomForestClassifier` for accurate predictions.
*   **Data Preprocessing**: Incorporates a `Normalizer` to scale input features, ensuring optimal model performance.
*   **Real-time Predictions**: Provides instant predictions, confidence scores, and personalized health tips.
*   **Error Handling**: Graceful error display for invalid inputs or backend issues.
*   **Containerization Ready**: Includes `requirements.txt` for easy dependency management and deployment.

## How to Run Locally

### Prerequisites
Make sure you have Python (3.7+) and `pip` installed.

### Setup
1.  **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    cd RevolutionizingLiverCare2
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    (Note: If you encounter issues with `scikit-learn` installation, ensure your `numpy` version is compatible, or try installing `scikit-learn` separately: `pip install scikit-learn`)
3.  **Ensure trained models exist**:
    Run the training script to generate `rf_acc_XX.pkl` and `normalizer.pkl` files. These files are essential for the Flask application.
    ```bash
    python Training/train_model.py
    ```

### Run the Flask App
1.  Navigate to the `Flask` directory:
    ```bash
    cd Flask
    ```
2.  Run the application:
    ```bash
    python app.py
    ```
3.  Open your web browser and go to `http://127.0.0.1:5000/` (or the address displayed in your terminal).

## Model Metrics
To get the latest model metrics (Accuracy and Classification Report) for your CV, run the following script:

```bash
python Training/train_model.py
```
The output in your terminal will provide these metrics.

## Test Data

### Test Case 1: Negative (Low Risk)
Expected Output: Not suffering from liver cirrhosis

```json
{
  "Age": 24,
  "Gender": 1,
  "Duration of alcohol consumption(years)": 0,
  "Quantity of alcohol consumption (quarters/day)": 0,
  "Hepatitis B infection": 0,
  "Hepatitis C infection": 0,
  "Diabetes Result": 0,
  "Obesity": 0,
  "Family history of cirrhosis/ hereditary": 0,
  "Platelet Count (lakhs/mm)": 2.6,
  "Total Bilirubin (mg/dl)": 0.8,
  "Direct (mg/dl)": 0.2,
  "Total Protein (g/dl)": 7.2,
  "Albumin (g/dl)": 4.5,
  "A/G Ratio": 1.4,
  "AL.Phosphatase (U/L)": 85,
  "SGOT/AST (U/L)": 28,
  "SGPT/ALT (U/L)": 32,
  "USG Abdomen (diffuse liver or not)": 0
}
```

### Test Case 2: Positive (High Risk)
Expected Output: Suffering from liver cirrhosis

```json
{
  "Age": 52,
  "Gender": 1,
  "Duration of alcohol consumption(years)": 18,
  "Quantity of alcohol consumption (quarters/day)": 4,
  "Hepatitis B infection": 1,
  "Hepatitis C infection": 0,
  "Diabetes Result": 1,
  "Obesity": 1,
  "Family history of cirrhosis/ hereditary": 1,
  "Platelet Count (lakhs/mm)": 0.9,
  "Total Bilirubin (mg/dl)": 4.2,
  "Direct (mg/dl)": 2.1,
  "Total Protein (g/dl)": 5.8,
  "Albumin (g/dl)": 2.7,
  "A/G Ratio": 0.7,
  "AL.Phosphatase (U/L)": 240,
  "SGOT/AST (U/L)": 160,
  "SGPT/ALT (U/L)": 120,
  "USG Abdomen (diffuse liver or not)": 1
}
```
