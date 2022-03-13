# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The used classifier is a Logistic Regression model from scikit-learn version 0.23.2. It uses the default hyperparameters and was trained by Janina Patzer.
## Intended Use

The algorithm aims to classify the salary of a person as above or below 50,000 USD based on several features.
## Training Data
The data is stored in a csv file and read as a dataframe. 80% of the data is used for training. The preprocessing consists of one hot encoding of categorical features and label binarization of the target variable 'salary'. 
## Evaluation Data
To be able to get a model prediction, the data preprocessing follows the steps mentioned in the previous paragraph. The test set makes up 20% of the whole dataset.
## Metrics
The metric used for evaluation are:
* Mean Accuracy: 0.79
* Precision: 0.73
* Recall: 0.27
* Fbeta: 0.39

## Ethical Considerations
From gained data insights there is a confirmed imbalance in the dataset (slice_tsting_scores.txt and clean_data.py).
## Caveats and Recommendations
The algorithm should be either adjusted to fit the imbalance of the dataset (e.g. with class weights or different solvers) or another algorithm should be tried to overcome this problem. The performance is not yet acceptable.
