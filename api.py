import os
import pickle
from datetime import datetime
from flask import Flask, send_file, jsonify, request, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "decision_tree.png")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
TEMP_MODEL_PATH = os.path.join(BASE_DIR, "temp_model.pkl")


def load_weather_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_temp_model():
    if not os.path.exists(TEMP_MODEL_PATH):
        return None
    with open(TEMP_MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ── Web UI ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Decision tree image ───────────────────────────────────────────────────────

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


# ── Weather condition prediction ──────────────────────────────────────────────

@app.route("/predict", methods=["GET"])
def predict_weather():
    data = load_weather_model()
    if not data:
        return jsonify({"error": "Model not found. Run decision_tree_model.py first."}), 404

    model         = data["model"]
    city_encoder  = data["city_encoder"]
    label_encoder = data["label_encoder"]
    available_cities = data["cities"]

    city          = request.args.get("city", "").strip().title()
    month         = request.args.get("month", datetime.now().month)
    temperature   = request.args.get("temperature")
    humidity      = request.args.get("humidity")
    precipitation = request.args.get("precipitation", 0)

    if not city:
        return jsonify({"error": "Missing param: city", "available_cities": available_cities}), 400
    if city not in available_cities:
        return jsonify({"error": f"Unknown city '{city}'", "available_cities": available_cities}), 400
    if temperature is None or humidity is None:
        return jsonify({"error": "Missing params: temperature, humidity"}), 400

    try:
        month         = int(month)
        temperature   = float(temperature)
        humidity      = float(humidity)
        precipitation = float(precipitation)
    except ValueError:
        return jsonify({"error": "Numeric params must be numbers"}), 400

    city_enc      = city_encoder.transform([city])[0]
    features      = [[city_enc, month, temperature, humidity, precipitation]]
    pred_enc      = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    predicted     = label_encoder.inverse_transform([pred_enc])[0]

    class_probs = {
        label_encoder.classes_[i]: round(float(p) * 100, 1)
        for i, p in enumerate(probabilities) if p > 0
    }
    class_probs = dict(sorted(class_probs.items(), key=lambda x: x[1], reverse=True))

    return jsonify({
        "input": {
            "city": city,
            "month": month,
            "temperature_celsius": temperature,
            "humidity_percent": humidity,
            "precipitation_mm": precipitation,
        },
        "predicted_weather": predicted,
        "confidence_percent": round(float(max(probabilities)) * 100, 1),
        "all_probabilities": class_probs,
    })


# ── Temperature prediction ────────────────────────────────────────────────────

@app.route("/predict-temperature", methods=["GET"])
def predict_temperature():
    data = load_temp_model()
    if not data:
        return jsonify({"error": "Temperature model not found. Run train_temperature_model.py first."}), 404

    model        = data["model"]
    city_encoder = data["city_encoder"]
    available_cities = data["cities"]

    city  = request.args.get("city", "").strip().title()
    month = request.args.get("month")

    if not city:
        return jsonify({"error": "Missing param: city", "available_cities": available_cities}), 400
    if city not in available_cities:
        return jsonify({"error": f"Unknown city '{city}'", "available_cities": available_cities}), 400
    if not month:
        return jsonify({"error": "Missing param: month (1-12)"}), 400

    try:
        month = int(month)
        if not 1 <= month <= 12:
            raise ValueError
    except ValueError:
        return jsonify({"error": "month must be an integer between 1 and 12"}), 400

    city_enc    = city_encoder.transform([city])[0]
    prediction  = model.predict([[city_enc, month]])[0]

    month_names = ["","January","February","March","April","May","June",
                   "July","August","September","October","November","December"]

    return jsonify({
        "input": {"city": city, "month": month_names[month]},
        "predicted_temperature_celsius": round(float(prediction), 2),
        "model": "Decision Tree Regressor",
        "model_r2_score": 0.849,
        "model_mae_celsius": 2.35,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
