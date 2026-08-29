"""
    USER → sends raw flight info
         ↓
API  → validates input
     → looks up engineered features from tables
     → feeds feature vector to model
     → returns prediction
         ↓ (packaged inside)
DOCKER → seals the API + models + lookups in a portable box
         ↓ (running on)
AWS  → gives the box a public address
"""

import pickle
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Loading the models and lookup tables on startup using
# asynccontextmanager to ensure they are loaded only once and shared across requests
# Usinf FastAPI lifespan events to manage the startup and shutdown of the application
@asynccontextmanager
async def lifespan(app: FastAPI):
    #Loading classification and regression models and lookup tables
    classification_model = xgb.XGBClassifier()
    regression_model = xgb.XGBRegressor()
    
    classification_model.load_model('models/classification_model.json')
    app.state.classification_model =  classification_model
    regression_model.load_model('models/regressor.json')
    app.state.regression_model = regression_model
    
    with open("models/feature_lookups.json", "r") as f:
        app.state.feature_lookups = json.load(f)
         
    # adding global rate from feature lookups to the app state for easy access
    app.state.global_rate = app.state.feature_lookups['global_delay_rate']
    
    yield
    
    # deleting at shutdown to free up memory
    del app.state.classification_model
    del app.state.regression_model
    del app.state.feature_lookups
    del app.state.global_rate
    

# Initializing the FastAPI app with the lifespan context manager
app = FastAPI(lifespan=lifespan)


# This model will be used to validate incoming requests to the API with pydantic. 
# It ensures that the input data adheres to the expected format and types.
class FlightInput(BaseModel):
    month: int
    day_of_week: int
    day_of_month: int
    dep_hour: int
    distance: float
    origin: str
    dest: str
    carrier: str
    
# build features function to transform the input data into a feature vector for the model
def build_features(flight: FlightInput) -> pd.DataFrame:
    route = f"{flight.origin}_{flight.dest}"
    
    features = {
        "month": flight.month,
        "day_of_month": flight.day_of_month,
        "day_of_week": flight.day_of_week,
        "distance": flight.distance,
        "dep_hour": flight.dep_hour,
        "origin_delay_rate": app.state.feature_lookups["origin_delay_rate"].get(flight.origin, app.state.global_rate),
        "dest_delay_rate": app.state.feature_lookups["dest_delay_rate"].get(flight.dest, app.state.global_rate),
        "carrier_delay_rate": app.state.feature_lookups["carrier_delay_rate"].get(flight.carrier, app.state.global_rate),
        "route_delay_rate": app.state.feature_lookups["route_delay_rate"].get(route, app.state.global_rate),
        "origin_pagerank": app.state.feature_lookups["origin_pagerank"].get(flight.origin, 0.001),
        "dest_pagerank": app.state.feature_lookups["dest_pagerank"].get(flight.dest, 0.001),
        "origin_flight_count": app.state.feature_lookups["origin_flight_count"].get(flight.origin, 100),
    }

    df = pd.DataFrame([features])
    # Force correct column order to match training
    expected_order = app.state.classification_model.get_booster().feature_names
    return df[expected_order]


@app.post("/predict")
async def predict(flight: FlightInput):
    X = build_features(flight)

    delay_prob = float(app.state.classification_model.predict_proba(X)[0, 1])
    is_delayed = bool(app.state.classification_model.predict(X)[0])
    delay_minutes = float(app.state.regression_model.predict(X)[0])

    return {
        "is_delayed": is_delayed,
        "delay_probability": round(delay_prob, 3),
        "predicted_delay_minutes": round(delay_minutes, 1),
        "input": flight.model_dump(),
    }