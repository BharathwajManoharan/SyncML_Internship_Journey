from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

model = keras.models.load_model('asl_model')
model.summary()

def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image, cmap='gray')
    plt.show()

def load_and_scale_image(image_path):
    image = keras.preprocessing.image.load_img(image_path, color_mode="grayscale", target_size=(28,28))
    return image

alphabet = "abcdefghiklmnopqrstuvwxy"

def predict_letter(image):
    image = keras.preprocessing.image.img_to_array(image)
    image = image.reshape(1,28,28,1)
    image = image/255
    prediction = model.predict(image)
    predicted_letter = alphabet[np.argmax(prediction)]
    return predicted_letter

image_path = "F:/Sync_Internship/Week4(4thMay)/asl_images/b.png"
show_image(image_path)
image = load_and_scale_image(image_path)
predicted_letter = predict_letter(image)
print(predicted_letter)

image_path = "F:/Sync_Internship/Week4(4thMay)/asl_images/a.png"
show_image(image_path)
image = load_and_scale_image(image_path)
predicted_letter = predict_letter(image)
print(predicted_letter)