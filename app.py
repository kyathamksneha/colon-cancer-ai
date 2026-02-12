import os
import numpy as np
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# ✅ Safe path loading
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "final_svm_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


# ✅ Home Page
@app.route('/')
def home():
    return render_template("index.html")


# ✅ Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Convert input to float
        features = [float(x) for x in request.form.values()]

        # Apply scaler
        final_features = scaler.transform(np.array(features).reshape(1, -1))

        # Predict
        prediction = model.predict(final_features)

        if prediction[0] == 1:
            result = "⚠️ High Risk of Colon Cancer"
        else:
            result = "✅ Low Risk of Colon Cancer"

        return render_template('index.html', prediction_text=result)

    except:
        return render_template(
            'index.html',
            prediction_text="⚠️ Please enter ONLY numeric values."
        )


if __name__ == "__main__":
    app.run(debug=True)
