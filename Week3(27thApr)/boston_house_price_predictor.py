import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Fit a decision tree regression model to the training data
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Fit a random forest regression model to the training data
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Fit a gradient boosting regression model to the training data
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Evaluate the performance of each model on the test data
print("Linear Regression R-squared:", linear_model.score(X_test, y_test))
print("Decision Tree Regression R-squared:", dt_model.score(X_test, y_test))
print("Random Forest Regression R-squared:", rf_model.score(X_test, y_test))
print("Gradient Boosting Regression R-squared:", gb_model.score(X_test, y_test))

# Save the best performing model to a file using Pickle
best_model = max(linear_model, dt_model, rf_model, gb_model, key=lambda model: model.score(X_test, y_test))
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
