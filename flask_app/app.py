from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
from .preprocessing_utility import normalize_text

# ===============================
# MLflow Tracking URI
# ===============================
mlflow.set_tracking_uri("http://54.91.250.234:5000/")

# ===============================
# Initialize Flask
# ===============================
app = Flask(__name__)

# ===============================
# Load model & vectorizer
# ===============================
model_name = "my_model"
model_alias = "champion"   # <-- use alias like 'champion' or 'staging'

model_uri = f"models:/{model_name}@{model_alias}"
model = mlflow.pyfunc.load_model(model_uri)

vectorizer_path = os.path.join("models", "vectorizer.pkl")
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# ===============================
# Routes
# ===============================
@app.route("/")
def home():
    return render_template("home.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]

    # preprocess
    text = normalize_text(text)

    # vectorize
    features = vectorizer.transform([text])
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # prediction
    result = model.predict(features_df)

    return render_template("home.html", result=result[0])

# ===============================
# Run locally
# ===============================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
