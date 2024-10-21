import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('resnet_cifar10_model.h5')

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_and_preprocess_image(img_path):
    """
    Load and preprocess an image for prediction.
    The input image is resized and normalized for ResNet50 model.
    """
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.0  # Normalize image
    return img_array

def predict_image(img_path):
    """
    Predict the class of an image using the trained ResNet model.
    """
    img_array = load_and_preprocess_image(img_path)
    
    # Perform the prediction
    predictions = model.predict(img_array)
    
    # Get the predicted class index and corresponding label
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_names[predicted_class_index]
    
    # Display the image and prediction result
    plt.imshow(image.load_img(img_path))
    plt.title(f'Prediction: {predicted_class_label}')
    plt.show()
    
    return predicted_class_label

if __name__ == "__main__":
    # Image file path to predict (replace with your image path)
    img_path = 'path_to_image_file.jpg'  # Update this to the path of the image you want to predict
    
    # Make prediction
    prediction = predict_image(img_path)
    print(f'The predicted class is: {prediction}')
