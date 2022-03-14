from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

import pandas as pd
import pytest

@pytest.fixture(scope="session")
def data():

    columns = [
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "salary",
    ]
    rows = [
        [39, "State-gov", 77516, "Bachelors", 13, "Never-married", "Adm-clerical", "Not-in-family", "White", "Male", 2174, 0, 40, "United-States", " <=50K"],
        [50, "Self-emp-not-inc", 83311, "Bachelors", 13, "Married-civ-spouse",  "Exec-managerial", "Husband", "White", "Male", 0, 0, 13,  "United-States", " <=50K"],
        [38, "Private", 215646, "HS-grad", 9,  "Divorced",  "Handlers-cleaners",  "Not-in-family", "White", "Male", 0, 0, 40,  "United-States", " <=50K"],
        [53, "Private", 234721, "11th", 7, "Married-civ-spouse",  "Handlers-cleaners", "Husband", "Black", "Male", 0, 0, 40,  "United-States", " <=50K"],
        [28, "Private", 338409, "Bachelors", 13, "Married-civ-spouse",  "Prof-specialty", "Wife", "Black", "Female", 0, 0, 40,  "Cuba", " <=50K]"],
        [37, "Private", 284582, "Masters", 14, "Married-civ-spouse",  "Exec-managerial", "Wife", "White", "Female", 0, 0, 40,  "United-States", " <=50K"],
        [49, "Private", 160187, "9th", 5, "Married-spouse-absent",  "Other-service",  "Not-in-family", "Black", "Female", 0, 0, 16,  "Jamaica", " <=50K"],
        [52, "Self-emp-not-inc", 209642, "HS-grad", 9, "Married-civ-spouse",  "Exec-managerial", "Husband", "White", "Male", 0, 0, 45,  "United-States", " >50K"],
        [31, "Private", 45781, "Masters", 14,  "Never-married",  "Prof-specialty",  "Not-in-family", "White", "Female", 14084, 0, 50,  "United-States"," >50K"],
        [42, "Private", 159449, "Bachelors", 13, "Married-civ-spouse",  "Exec-managerial", "Husband", "White", "Male", 5178, 0, 40,  "United-States", " >50K"]
    ]
    
    df = pd.DataFrame(rows, columns=columns)

    return df

def test_data_shape(data):

    assert data.isna().sum().sum() == 0

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

    cat_cols_train = [cat_features[idx] for idx in range(len(cat_features)-1) if X_train[idx].dtype.name == 'category']
    cat_cols_test =  [cat_features[idx] for idx in range(len(cat_features)-1) if x_test[idx].dtype.name == 'category']
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

