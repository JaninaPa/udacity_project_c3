from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import numpy as np

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    clf = LogisticRegression()
    clf.fit(X_train,y_train)

    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def slice_testing(model,data,column,categorical_columns,lb,encoder,label):
    """ Run performance tests on slices of data.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    data : pd.DataFrame
        Data used for prediction.
    column: str
        Name of column to be sliced.
    categorical_columns: list
        List of categorical features in data.
    lb: LabelBinarizer object
        Label binarizer.
    encodrr: OneHotEncoder object
        One Hot Encoder.
    label: str
        Name of target column.
    
    
    Returns
    -------
    performances: list
        List of dictionaries containing column value, mean accuracy, precision, recall and fbeta scores.
    """

    performances = []
    for value in data[column].unique():
        df_temp = data[data[column] == value]
        y = df_temp[label]
        X = df_temp.drop([label],axis=1)

        # Add processing steps from process_data

        X_continuous = X.drop(*[categorical_columns], axis=1)
        X_categorical = X[categorical_columns].values
        X_categorical = encoder.transform(X_categorical)
        y = lb.transform(y.values).ravel()
        X = np.concatenate([X_continuous, X_categorical], axis=1)

        predictions = inference(model,X)

        # Evaluate model

        mean_acc = model.score(X,y)
        precision, recall, fbeta = compute_model_metrics(y, predictions)

        performances.append({"Value":value,
                        "Mean Accuracy": mean_acc,
                        "Precision": precision,
                        "Recall": recall,
                        "Fbeta":fbeta})

    return performances
