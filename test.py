import tensorflow as tf
import numpy as np
from PIL import Image

loaded_model = tf.keras.models.load_model('E:/Models/my_first_model')   #Replace the path with the path you saved
custom_image = Image.open('E:/Downloads/archive/testSample/img_80.jpg')  # Replace with the actual image file path
custom_image = custom_image.resize((28, 28))
custom_image = custom_image.convert('L')
custom_image = np.array(custom_image) / 255.0
custom_image = custom_image.reshape(1, 784)  # Reshape the image to match the model's input shape

predictions = loaded_model.predict(custom_image)
predicted_label = np.argmax(predictions)

print(f"Predicted label: {predicted_label}")
