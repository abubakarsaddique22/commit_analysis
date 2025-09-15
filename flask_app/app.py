from flask import Flask, render_template, request
import mlflow
import pickle
import pandas as pd
import numpy as np
import re, string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================
# Text preprocessing functions
# =========================
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

# =========================
# MLflow setup
# =========================
mlflow.set_tracking_uri("http://54.91.250.234:5000/")

app = Flask(__name__)

# Load latest production model
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_name = "my_model"
model_version = get_latest_model_version(model_name)
model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

# Load vectorizer
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# Label mapping (from LabelEncoder)
label_mapping = {
    0:'anger', 1:'boredom', 2:'empty', 3:'enthusiasm', 4:'fun',
    5:'happiness', 6:'hate', 7:'love', 8:'neutral', 9:'relief',
    10:'sadness', 11:'surprise', 12:'worry'
}

# =========================
# Routes
# =========================
@app.route('/')
def home():
    return render_template('home.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # Preprocess the text
    text = normalize_text(text)

    # Transform text to features
    features = vectorizer.transform([text])
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # Predict numeric label
    pred_num = model.predict(features_df)[0]

    # Convert numeric label to sentiment text
    pred_text = label_mapping[pred_num]

    # Render template with the sentiment
    return render_template('home.html', result=pred_text)

# =========================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
