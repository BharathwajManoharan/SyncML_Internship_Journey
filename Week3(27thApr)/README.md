## Boston House Price Prediction

This project uses machine learning algorithms to predict the median value of owner-occupied homes in Boston, Massachusetts. The dataset used to train the model contains information such as crime rate, average number of rooms per dwelling, and pupil-teacher ratio by town.

## Installation

To use this project, follow the steps below:

* Clone the repository.
* Install the required packages using pip install -r requirements.txt.

## Data Preprocessing

The dataset is preprocessed to handle missing values and feature scaling. This is done in the preprocess.py script.

## Model Training and Evaluation

The following machine learning algorithms are used to predict the median value of owner-occupied homes:

Linear Regression
Decision Tree Regression
Random Forest Regression
Gradient Boosting Regression

The model training and evaluation is done in the train_eval.py script. The script trains each algorithm on the preprocessed data and evaluates their performance using cross-validation. The best performing algorithm is saved as a .pkl file.

## Prediction

To predict the median value of a house, run the predict.py script and input the values of the features. The script will load the best performing model from the .pkl file and output the predicted value.

## Credits

The dataset used in this project is the Boston Housing dataset, which was originally published by Harrison, D. and Rubinfeld, D.L. in 1978.




