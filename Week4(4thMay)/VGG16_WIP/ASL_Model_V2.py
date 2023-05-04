from tensorflow import keras
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import (Dense,MaxPool2D,Flatten,Dropout)
from keras.applications.vgg16 import VGG16 

# Load the VGG16 model without the top layer
vggmodel = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Load the pre-trained weights for the first 13 layers only
weights_path = './vgg16_weights_tf_dim_ordering_tf_kernels.h5'
num_layers_to_load = 13
for i in range(num_layers_to_load):
    vggmodel.layers[i].set_weights(vggmodel.layers[i].get_weights())
       
# Load in our data from CSV files
train_df = pd.read_csv("F:/Sync_Internship/Week4(4thMay)/asl_data/sign_mnist_train.csv")
valid_df = pd.read_csv("F:/Sync_Internship/Week4(4thMay)/asl_data/sign_mnist_valid.csv")

# Separate out our target values
y_train = train_df['label']
y_valid = valid_df['label']
del train_df['label']
del valid_df['label']

# Separate our our image vectors
x_train = train_df.values
y_valid = valid_df.values

# Resize the images
x_train_resized = []
for img in x_train:
    img = img.reshape((28, 28))
    img = Image.fromarray(img).resize((32, 32))
    img = np.array(img)
    x_train_resized.append(img)

y_valid_resized = []
for img in y_valid:
    img = img.reshape((28, 28))
    img = Image.fromarray(img).resize((32, 32))
    img = np.array(img)
    y_valid_resized.append(img)

# Convert to numpy arrays
x_train_resized = np.array(x_train_resized)
y_valid_resized = np.array(y_valid_resized)

# Reshape the image data for the convolutional network
x_train_resized = x_train_resized.reshape(-1,32,32,1)
y_valid_resized = y_valid_resized.reshape(-1,32,32,1)


# Turn our scalar targets into binary categories
num_classes = 24
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

# Normalize our image data
x_train_resized = x_train_resized / 255
y_valid_resized = y_valid_resized / 255


# Freeze the layers in the pre-trained model
for layer in vggmodel.layers:
    layer.trainable = False

# Define the model architecture
x = Flatten()(vggmodel.output)
x = Dense(units=512, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(units=num_classes, activation="softmax")(x)
model = Model(inputs=vggmodel.input, outputs=output)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,  
    zoom_range=0.1,  
    width_shift_range=0.1, 
    height_shift_range=0.1,  
    horizontal_flip=True,  
    vertical_flip=False, 
) 

# Batch Size
batch_size = 32
img_iter = datagen.flow(x_train_resized, y_valid_resized, batch_size=batch_size)

x, y = img_iter.next()
fig, ax = plt.subplots(nrows=4, ncols=8)
for i in range(batch_size):
    image = x[i]
    ax.flatten()[i].imshow(np.squeeze(image))
plt.show()

# Fitting the Data to the Generator
datagen.fit(x_train_resized )

# Compiling the Model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
model.fit(img_iter,
          epochs=10,
          steps_per_epoch=len(x_train_resized )/batch_size,
          validation_data=(x_train_resized , y_valid_resized ))

# Saving the Model
model.save('asl_model_V2')