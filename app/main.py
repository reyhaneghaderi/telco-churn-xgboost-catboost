import joblib
import pandas as pd

from fastapi import FastAPI

from app.schemas import CustomerFeatures


# Create FastAPI application
app = FastAPI()


# Load saved model
bundle = joblib.load("models/churn_xgb_bundle.joblib")

model = bundle["pipeline"]
threshold = bundle["threshold"]


# Home page
@app.get("/")
def home():
    return {"message": "Telco Churn API is working"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }



# Prediction
@app.post("/predict")
def predict(customer: CustomerFeatures):

    # Convert customer data to dictionary
    customer_data = customer.model_dump()

    # Convert dictionary to DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Get churn probability
    probability = model.predict_proba(customer_df)[0][1]

    # Convert probability to prediction
    if probability >= threshold:
        prediction = "Churn"
    else:
        prediction = "No Churn"

    return {
        "prediction": prediction,
        "churn_probability": float(probability)
    }
