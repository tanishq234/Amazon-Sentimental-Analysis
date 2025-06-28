from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import re
from io import BytesIO
import os
import traceback

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64

# Ensure NLTK data is downloaded
import nltk
nltk.download('stopwords')

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Load models only once at startup
try:
    import xgboost as xgb
    predictor = xgb.XGBClassifier()
    predictor.load_model("Models/model_xgb.json")
    scaler = pickle.load(open("Models/scaler.pkl", "rb"))
    cv = pickle.load(open("Models/countVectorizer.pkl", "rb"))
except Exception as e:
    print("❌ Error loading model files:", e)
    predictor = scaler = cv = None


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")


@app.route("/test", methods=["GET"])
def test():
    return "✅ Test request received. Service is running."


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure model is loaded
        if not all([predictor, scaler, cv]):
            raise RuntimeError("Model files not loaded.")

        # CSV Upload
        if "file" in request.files:
            file = request.files["file"]
            data = pd.read_csv(file)

            if "Sentence" not in data.columns:
                return jsonify({"error": "CSV must contain a 'Sentence' column."}), 400

            predictions, graph = bulk_prediction(data)

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv"
            )

            # Send base64 graph in headers
            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(graph.getbuffer()).decode("utf-8")
            return response

        # Single text input
        elif request.is_json and "text" in request.json:
            text_input = request.json["text"]
            sentiment = single_prediction(text_input)
            return jsonify({"prediction": sentiment})

        else:
            return jsonify({"error": "No valid input received."}), 400

    except Exception as e:
        print("❌ Exception in /predict:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def single_prediction(text_input):
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input).lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    corpus = [" ".join(review)]

    X = cv.transform(corpus).toarray()
    X_scaled = scaler.transform(X)
    pred = predictor.predict_proba(X_scaled).argmax(axis=1)[0]

    return "Positive" if pred == 1 else "Negative"


def bulk_prediction(data):
    stemmer = PorterStemmer()
    corpus = []

    for sentence in data["Sentence"]:
        review = re.sub("[^a-zA-Z]", " ", str(sentence)).lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        corpus.append(" ".join(review))

    X = cv.transform(corpus).toarray()
    X_scaled = scaler.transform(X)
    preds = predictor.predict_proba(X_scaled).argmax(axis=1)
    data["Predicted sentiment"] = ["Positive" if p == 1 else "Negative" for p in preds]

    # Save predictions to CSV
    csv_output = BytesIO()
    data.to_csv(csv_output, index=False)
    csv_output.seek(0)

    return csv_output, generate_pie_chart(data)


def generate_pie_chart(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01,) * len(tags)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png", bbox_inches="tight")
    plt.close()
    graph.seek(0)
    return graph


if __name__ == "__main__":
    port=int(os.environ.get("PORT",8000))
    app.run(host='0.0.0.0',port=port)
