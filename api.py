import os
from flask import Flask, send_file, jsonify

app = Flask(__name__)

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "decision_tree.png")


@app.route("/")
def index():
    return jsonify({
        "service": "Decision Tree API",
        "endpoints": {
            "GET /decision-tree/image": "Download the decision tree graph as a PNG file"
        }
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
