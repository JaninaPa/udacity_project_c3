from fastapi import FastAPI
from typing import List
from pydantic import Field, BaseModel
import pickle
from ml.model import inference
import pandas as pd
import numpy as np
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


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
                "age": [42],
                "workclass": ["Private"],
                "fnlgt": [159449],
                "education": ["Bachelors"],
                "education-num": [13],
                "marital-status": ["Married-civ-spouse"],
                "occupation": ["Exec-managerial"],
                "relationship": ["Husband"],
                "race": ["White"],
                "sex": ["Male"],
                "capital-gain": [5178],
                "capital-loss": [0],
                "hours-per-week": [40],
                "native-country": ["United-States"]
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
    df.columns = df.columns.str.replace("_", "-")

    X_categorical = df[cat_features].values
    X_continuous = df.drop(*[cat_features], axis=1)

    encoder = pickle.load(open("rf_enc_fastapi.sav", "rb"))
    X_categorical = encoder.transform(X_categorical)
    X = np.concatenate([X_continuous, X_categorical], axis=1)

    classifier = pickle.load(open("rf_model_fastapi.sav", "rb"))

    predictions = inference(classifier, X)

    return {"results": predictions.tolist()}
