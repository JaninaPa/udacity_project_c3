from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

import pandas as pd
import pytest

@pytest.fixture(scope="session")
def data():

    df = pd.read_csv("../census.csv")

    return df.head(10)

def test_data_shape(data):

    assert data.isna().sum() == 0

def test_process_data(data):

    # Split data in train and test set

    train, test = train_test_split(data, test_size=0.20)

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
    # Process data
    X_train, y_train, encoder, lb = process_data(
        train, 
        categorical_features=cat_features, 
        label="salary", 
        training=True
    )

    x_test, y_test, encoder_test, lb_test = process_data(
        test,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        label="salary",
        training=False,
    )

    cat_cols_train = [col for col in cat_features if X_train[col].values.dtype.name == 'category']
    cat_cols_test =  [col for col in cat_features if x_test[col].values.dtype.name == 'category']
    cat_cols = cat_cols_train+cat_cols_test

    assert len(cat_cols) == 0
    assert y_train.unqiue() == [0,1]
    assert y_test.unqiue() == [0,1]

def test_train_model(data):
    # Split data in train and test set

    train, test = train_test_split(data, test_size=0.20)

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
    # Process data
    X_train, y_train, encoder, lb = process_data(
        train, 
        categorical_features=cat_features, 
        label="salary", 
        training=True
    )

    x_test, y_test, encoder_test, lb_test = process_data(
        test,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        label="salary",
        training=False,
    )

    # Fit classifier

    model = train_model(X_train,y_train)

    assert model

def test_inference(data):

    # Split data in train and test set

    train, test = train_test_split(data, test_size=0.20)

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
    # Process data
    X_train, y_train, encoder, lb = process_data(
        train, 
        categorical_features=cat_features, 
        label="salary", 
        training=True
    )

    x_test, y_test, encoder_test, lb_test = process_data(
        test,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        label="salary",
        training=False,
    )

    # Fit classifier

    model = train_model(X_train,y_train)
    predictions = inference(model,x_test)

    # Evaluate model

    mean_acc = model.score(x_test,y_test)
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)

    assert mean_acc != 0
    assert isinstance(precision,float)
    assert isinstance(recall,float)
    assert isinstance(fbeta,float)

