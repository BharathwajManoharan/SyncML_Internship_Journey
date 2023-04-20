## Real-time Face Mask Detection
This project uses deep learning to detect if a person is wearing a face mask in real-time. The model is built using TensorFlow and Keras and uses MobileNetV2 as the base architecture. The dataset used to train the model was created by manually collecting images of people wearing and not wearing masks from various sources.

## Installation
To use this project, follow the steps below:

* Clone the repository
* Install the required packages using pip install -r requirements.txt

The following packages will be installed:

* tensorflow
* numpy
* imutils
* keras
* opencv-python
* scipy
* matplotlib

## Train the model
Run train.py to train the model. This script will preprocess the dataset, create the model, train it on the preprocessed data, and save the trained model as mask_detection_model.h5. You can modify the hyperparameters in config.py.

## Detect face masks in real-time
Run detect_mask.py to launch the real-time face mask detection application. This script will load the trained model from mask_detection_model.h5 and use it to predict whether a person is wearing a face mask or not in real-time.

## Credits
The dataset used to train the model was created by Prajna Bhandary and Adrian Rosebrock. The base architecture used in this project is MobileNetV2.
