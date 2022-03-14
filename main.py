from fastapi import FastAPI
from typing import List
from pydantic import Field, BaseModel
import pickle
from ml.model import inference
import pandas as pd
import numpy as np

class RequestBody(BaseModel):
    age: List[int]
    workclass: List[str]
    fnlgt: List[int]
    education: List[str]
    education_num: List[int] = Field(alias="education-num")
    marital_status: List[str] = Field(alias="marital-status")
    occupation: List[str]
    relationship: List[str]
    race: List[str]
    sex: List[str]
    capital_gain: List[int] = Field(alias="capital-gain")
    capital_loss: List[int] = Field(alias="capital-loss")
    hours_per_week: List[int] = Field(alias="hours-per-week")
    native_country: List[str] = Field(alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                "age":42, 
                "workclass":"Private",
                "fnlgt": 159449, 
                "education":"Bachelors", 
                "education_num":13, 
                "marital_status":"Married-civ-spouse",  
                "occupation":"Exec-managerial",
                "relationship":"Husband", 
                "race":"White", 
                "sex":"Male", 
                "capital_gain":5178, 
                "capital_loss":0, 
                "hours_per_week": 40,  
                "native_country":"United-States"
            }
        }

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Welcome!"}

@app.post("/model_inference/")
async def predict(request_body: RequestBody):

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    df = pd.DataFrame(request_body.dict())
    df.columns = df.columns.str.replace("_","-")

    X_categorical = df[cat_features].values
    X_continuous = df.drop(*[cat_features], axis=1)

    encoder = pickle.load(open("rf_enc_fastapi.sav", "rb"))
    X_categorical = encoder.transform(X_categorical)
    X = np.concatenate([X_continuous, X_categorical], axis=1)

    classifier = pickle.load(open("rf_model_fastapi.sav", "rb"))
    
    predictions = inference(classifier, X)

    return {"results": predictions.tolist()}