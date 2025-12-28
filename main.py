from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import json
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgb_flight_delay_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model_features.json")

# Load model
model = joblib.load(MODEL_PATH)

# Load feature list (must be a list)
with open(FEATURES_PATH, "r") as f:
    model_features = json.load(f)
if not isinstance(model_features, list):
    raise ValueError("model_features.json must contain a JSON list of column names.")

def parse_value(v):
    """Convert form string value to appropriate type (bool or float)."""
    if isinstance(v, bool):
        return v
    if v is None or str(v).strip() == "":
        return 0  # default for missing numeric inputs
    s = str(v).strip()
    if s.lower() in ("true", "1", "yes", "y", "t"):
        return True
    if s.lower() in ("false", "0", "no", "n", "f"):
        return False
    # try integer then float
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return s  # fallback (string); model.reindex will fill with 0 for unknowns

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Accept form or JSON
    if request.is_json:
        body = request.get_json()
        row = body.get("row", body) if isinstance(body, dict) else {}
    else:
        row = dict(request.form)

    # Parse values
    parsed = {}
    for k, v in row.items():
        parsed[k] = parse_value(v)

    # Build DataFrame aligned to model features
    X = pd.DataFrame([ {c: parsed.get(c, 0) for c in model_features} ])

    # Ensure dtypes roughly match (optional)
    # Predict
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0,1])

    # Return JSON if requested, otherwise simple string
    if request.is_json:
        return jsonify({"prediction": pred, "probability": proba})
    return f"Prediction: {pred} — Probability: {proba:.2f}"

if __name__ == "__main__":
    # host=0.0.0.0 if you want external access; change port if needed
    app.run(debug=True, host="127.0.0.1", port=5000)
