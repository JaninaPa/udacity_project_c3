# Script to train machine learning model.

from cv2 import mean
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from ml.data import process_data
from ml.model import *

# Load in  data

data = pd.read_csv("census_cleaned.csv")

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

print("###### Performance ######")
print("Mean Accuracy: ", mean_acc)
print("Precision:     ", precision)
print("Recall:        ", recall)
print("Fbeta:         ", fbeta)

# Save model

model_file = 'census_lr_model.sav'
encoder_file = 'census_lr_enc.sav'

pickle.dump(model, open(model_file, 'wb'))
pickle.dump(model, open(encoder_file, 'wb'))