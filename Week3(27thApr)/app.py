from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model from the file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a Flask web application
app = Flask(__name__)

# Define a route to the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user's input values from the form
    CRIM = float(request.form['CRIM'])
    ZN = float(request.form['ZN'])
    INDUS = float(request.form['INDUS'])
    NOX = float(request.form['NOX'])
    RM = float(request.form['RM'])
    AGE = float(request.form['AGE'])
    DIS = float(request.form['DIS'])
    RAD = float(request.form['RAD'])
    TAX = float(request.form['TAX'])
    PTRATIO = float(request.form['PTRATIO'])
    B = float(request.form['B'])
    LSTAT = float(request.form['LSTAT'])

    # Create a NumPy array from the user's input values
    input_data = np.array([[CRIM, ZN, INDUS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])

    # Use the trained model to make a prediction on the user's input values
    predicted_price = model.predict(input_data)[0]

    # Render the prediction result on the prediction page
    return render_template('predict.html', predicted_price=predicted_price)

# Run the web application
if __name__ == '__main__':
    app.run(debug=True)
