## Sign Language Recognition (ASL)
This project is a part of the Sync Intern #ML Internship program. The aim of the project is to recognize the American Sign Language (ASL) alphabet using a Convolutional Neural Network (CNN). The project consists of two codes:

# ASL_Model Generation: 
This code generates a CNN model to recognize the ASL alphabet. The model is trained using the sign_mnist_train.csv and sign_mnist_valid.csv datasets, which contain images of the ASL alphabet in grayscale. The model is saved as 'asl_model'. Data augmentation techniques are used to increase the size of the dataset.

# Hand Detection:
using model generated from above code: This code loads the 'asl_model' and uses it to recognize the ASL alphabet in images of hands. The alphabet is predicted by first preprocessing the image to match the format used in training the model.

# Prerequisites
The project requires Python 3.x and the following libraries:

* tensorflow
* keras
* pandas
* numpy
* matplotlib
These can be installed using pip.

# Usage
* Run ASL_Model Generation.py to generate and save the ASL recognition model.
* Run Hand Detection using model generated from above code.py to load the saved model and recognize the ASL alphabet in images of hands. You can modify the file to test your own images.

# Acknowledgments
The project uses the sign_mnist_train.csv and sign_mnist_valid.csv datasets from Kaggle (https://www.kaggle.com/datamunge/sign-language-mnist)
