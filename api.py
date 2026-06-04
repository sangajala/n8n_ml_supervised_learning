import os
import pickle
from datetime import datetime
from flask import Flask, send_file, jsonify, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
IMAGE_PATH = os.path.join(BASE_DIR, "decision_tree.png")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@app.route("/")
def index():
    data = load_model()
    cities = data["cities"] if data else []
    return jsonify({
        "service": "UK Weather Decision Tree API",
        "endpoints": {
            "GET /decision-tree/image": "Download the decision tree graph as PNG",
            "GET /predict?city=London&month=6&temperature=18&humidity=75&precipitation=2": "Predict weather condition"
        },
        "available_cities": cities
    })


@app.route("/decision-tree/image", methods=["GET"])
def download_decision_tree():
    if not os.path.exists(IMAGE_PATH):
        return jsonify({"error": "Image not found. Run decision_tree_model.py first."}), 404
    return send_file(
        IMAGE_PATH,
        mimetype="image/png",
        as_attachment=True,
        download_name="decision_tree.png"
    )


@app.route("/predict", methods=["GET"])
def predict():
    data = load_model()
    if not data:
        return jsonify({"error": "Model not found. Run decision_tree_model.py first."}), 404

    model = data["model"]
    city_encoder = data["city_encoder"]
    label_encoder = data["label_encoder"]
    available_cities = data["cities"]

    # Get params
    city = request.args.get("city", "").strip().title()
    month = request.args.get("month", datetime.now().month)
    temperature = request.args.get("temperature")
    humidity = request.args.get("humidity")
    precipitation = request.args.get("precipitation", 0)

    # Validate city
    if not city:
        return jsonify({"error": "Missing required param: city", "available_cities": available_cities}), 400
    if city not in available_cities:
        return jsonify({"error": f"Unknown city '{city}'", "available_cities": available_cities}), 400

    # Validate numeric params
    if temperature is None or humidity is None:
        return jsonify({"error": "Missing required params: temperature, humidity"}), 400

    try:
        month = int(month)
        temperature = float(temperature)
        humidity = float(humidity)
        precipitation = float(precipitation)
    except ValueError:
        return jsonify({"error": "month, temperature, humidity, precipitation must be numbers"}), 400

    # Encode and predict
    city_enc = city_encoder.transform([city])[0]
    features = [[city_enc, month, temperature, humidity, precipitation]]
    prediction_enc = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    predicted_class = label_encoder.inverse_transform([prediction_enc])[0]

    # Build probability breakdown
    class_probs = {
        label_encoder.classes_[i]: round(float(p) * 100, 1)
        for i, p in enumerate(probabilities)
        if p > 0
    }
    class_probs = dict(sorted(class_probs.items(), key=lambda x: x[1], reverse=True))

    return jsonify({
        "input": {
            "city": city,
            "month": month,
            "temperature_celsius": temperature,
            "humidity_percent": humidity,
            "precipitation_mm": precipitation
        },
        "predicted_weather": predicted_class,
        "confidence_percent": round(float(max(probabilities)) * 100, 1),
        "all_probabilities": class_probs
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
