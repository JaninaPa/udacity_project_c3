# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The used classifier is a Decision Tree model from scikit-learn version 1.0.2. It uses the default hyperparameters and was trained by Janina Patzer.
## Intended Use

The algorithm aims to classify the salary of a person as above or below 50,000 USD based on several features.
## Training Data
The data is stored in a csv file and was obtained from the repository https://github.com/udacity/nd0821-c3-starter-code.git . 80% of the data is used for training. The preprocessing consists of one hot encoding of categorical features and label binarization of the target variable 'salary'. 
## Evaluation Data
To be able to get a model prediction, the data preprocessing follows the steps mentioned in the previous paragraph. The test set makes up 20% of the whole dataset.
## Metrics
The metrics used for evaluation:
* Mean Accuracy: 0.81
* Precision: 0.62
* Recall: 0.63
* Fbeta: 0.63

## Ethical Considerations
From gained data insights there is a confirmed imbalance in the dataset (slice_testing_scores.txt and clean_data.py).
## Caveats and Recommendations
The algorithm should be either adjusted to fit the imbalance of the dataset (e.g. with class weights) or another algorithm should be tried to overcome this problem. The performance is not yet sufficient.