from data_collector import Datacollector
from resnet import ResidualBlock

import tensorflow as tf
import keras

def main():
    model = keras.models.load_model('models/resnet_model_1.h5', custom_objects={'ResidualBlock': ResidualBlock})
    
    # Sample image for prediction
    sample_image_path = "Data/Animalia/Animalia_image_0999.jpg"  # Replace with your image path
    sample_image = tf.keras.preprocessing.image.load_img(sample_image_path, target_size=(224, 224))
    sample_image = tf.keras.preprocessing.image.img_to_array(sample_image)
    sample_image = tf.expand_dims(sample_image, axis=0)  # Add batch dimension
    sample_image = sample_image / 255.0  # Normalize the image
    
    # Make prediction
    predictions = model.predict(sample_image)
    predicted_class = tf.argmax(predictions[0]).numpy()
    
    class_names = ["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"] 
    
    predicted_label = class_names[predicted_class]
    print(f"Predicted class: {predicted_label}")
    
if __name__ == "__main__":
    main()