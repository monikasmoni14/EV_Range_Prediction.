from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained ML model
model = joblib.load("models/ev_range_model.pkl")

# Load preprocessed dataset
data = pd.read_csv(
    "data/Electric_Vehicle_Population_Data_Preprocessed.csv"
)

# Get unique Make and Model values
makes = sorted(
    data["Make"].dropna().unique().tolist()
)

models = sorted(
    data["Model"].dropna().unique().tolist()
)


@app.route("/")
def home():

    return render_template(
        "index.html",
        makes=makes,
        models=models
    )


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from HTML form
    model_year = int(request.form["model_year"])
    make = request.form["make"]
    vehicle_model = request.form["vehicle_model"]
    vehicle_type = request.form["vehicle_type"]
    cafv = request.form["cafv"]
    base_msrp = float(request.form["base_msrp"])

    # Create input dataframe
    input_data = pd.DataFrame([{
        "Model Year": model_year,
        "Make": make,
        "Model": vehicle_model,
        "Electric Vehicle Type": vehicle_type,
        "Clean Alternative Fuel Vehicle (CAFV) Eligibility": cafv,
        "Base MSRP": base_msrp
    }])

    # Predict electric range
    prediction = model.predict(input_data)[0]

    prediction = round(prediction, 2)

    # Show result
    return render_template(
        "result.html",
        prediction=prediction,
        model_year=model_year,
        make=make,
        vehicle_model=vehicle_model,
        vehicle_type=vehicle_type,
        cafv=cafv,
        base_msrp=base_msrp
    )


if __name__ == "__main__":
    app.run(debug=True)