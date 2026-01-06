import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask("house-price")

with open("model.bin", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
dv = bundle["dv"]


@app.route("/predict", methods=["POST"])
def predict():
    house = request.get_json(silent=True)

    if house is None:
        return jsonify({"error": "empty json"}), 400

    X = dv.transform([house])
    pred_log = model.predict(X)[0]
    pred_price = float(np.expm1(pred_log))

    return jsonify({"predicted_price_in_rupees": pred_price})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=False, use_reloader=False)