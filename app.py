# app.py
from flask import Flask, render_template, request
import joblib
import pandas as pd

# ------------------------------------------------------------
# 1️⃣  Load the fitted pipeline once at start‑up
# ------------------------------------------------------------
PIPE_PATH = "delivery_time_pipeline.pkl"
pipe = joblib.load(PIPE_PATH)

# ------------------------------------------------------------
# 2️⃣  List of feature names in the exact order used in training
# ------------------------------------------------------------
FEATURES = [
    "Delivery_person_Age", "Delivery_person_Ratings", "Weather_conditions",
    "Road_traffic_density", "Vehicle_condition", "Type_of_order",
    "Type_of_vehicle", "multiple_deliveries", "Festival", "City",
    "City_code", "day", "month", "quarter", "year", "day_of_week",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end",
    "is_year_start", "is_year_end", "is_weekend", "order_prepare_time",
    "distance"
]

# ------------------------------------------------------------
# 3️⃣  Flask app & routes
# ------------------------------------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    pred = None
    if request.method == "POST":
        # Collect inputs as floats (or ints) in the correct order
        values = [float(request.form.get(feat, 0)) for feat in FEATURES]
        X_new  = pd.DataFrame([values], columns=FEATURES)
        pred   = float(pipe.predict(X_new)[0])

    # ⬇️  Pass FEATURES into the template as 'features'
    return render_template("index.html", pred=pred, features=FEATURES)

# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
