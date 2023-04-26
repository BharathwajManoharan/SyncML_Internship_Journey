import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load Boston dataset from Scikit-Learn library
columns = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE',
           'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'PRICE']
boston = pd.read_csv('F:/Sync_Internship/Week3(27thApr)/housing.csv', delimiter=r"\s+", names=columns)

# Split the data into features and target
X = boston.drop('PRICE', axis=1)
y = boston['PRICE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a file using Pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
