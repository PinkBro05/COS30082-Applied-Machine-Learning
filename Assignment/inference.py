from ultis.data_collector import Datacollector
from models.resnet_from_scratch import ResidualBlock

import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import random
import os

def predict_single_image(model, image_path, class_names):
    """
    Make a prediction on a single image
    
    Args:
        model: Loaded Keras model
        image_path: Path to the image file
        class_names: List of class names
    """
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    confidence = float(predictions[0, predicted_class])
    
    # Get predicted label
    predicted_label = class_names[predicted_class]
    
    # Return results
    return {
        'class': predicted_label,
        'confidence': confidence,
        'predictions': {class_names[i]: float(predictions[0, i]) for i in range(len(class_names))}
    }

def visualize_prediction(image_path, prediction_result):
    """
    Visualize the prediction result with a bar chart of probabilities
    """
    # Load and display image
    img = plt.imread(image_path)
    
    plt.figure(figsize=(12, 6))
    
    # Plot the image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Predicted: {prediction_result['class']}\nConfidence: {prediction_result['confidence']:.2%}")
    plt.axis('off')
    
    # Plot the prediction probabilities
    plt.subplot(1, 2, 2)
    classes = list(prediction_result['predictions'].keys())
    probabilities = list(prediction_result['predictions'].values())
    
    # Sort by probability
    sorted_indices = np.argsort(probabilities)[::-1]
    classes = [classes[i] for i in sorted_indices]
    probabilities = [probabilities[i] for i in sorted_indices]
    
    # Horizontal bar chart
    plt.barh(range(len(classes)), probabilities)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel('Probability')
    plt.title('Class Probabilities')
    plt.tight_layout()
    plt.show()

def get_class_names_from_directory(data_path="Data/train"):
    """Get class names in alphabetical order as used during training"""
    class_names = sorted([item.name for item in os.scandir(data_path) 
                         if item.is_dir() and not item.name.startswith('.')])
    return class_names

def main():
    # Load model
    model = keras.models.load_model('models/best_resnet_model_balanced.h5')
    
    # Get class names in the same order as used during training
    class_names = get_class_names_from_directory("Data/train")
    print(f"Using class names in alphabetical order: {class_names}")
    
    # Choose a random test image to predict
    test_dir = pathlib.Path("Data/test")
    
    # Get all class folders
    class_folders = [f for f in test_dir.iterdir() if f.is_dir()]
    
    # Choose a random class folder
    random_class_folder = random.choice(class_folders)
    class_name = random_class_folder.name
    
    # Get all image files in the class folder
    image_files = list(random_class_folder.glob('*.jpg'))
    if not image_files:
        image_files = list(random_class_folder.glob('*.jpeg'))
    if not image_files:
        image_files = list(random_class_folder.glob('*.png'))
    
    # Choose a random image
    if image_files:
        random_image = random.choice(image_files)
        image_path = str(random_image)
        
        print(f"\nSelected random test image: {image_path}")
        print(f"Actual class: {class_name}")
        
        # Make prediction
        result = predict_single_image(model, image_path, class_names)
        
        # Print results
        print(f"Predicted class: {result['class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nTop 3 predictions:")
        top_predictions = sorted(result['predictions'].items(), key=lambda x: x[1], reverse=True)[:3]
        for class_name, confidence in top_predictions:
            print(f"- {class_name}: {confidence:.2%}")
        
        # Visualize the prediction
        visualize_prediction(image_path, result)
    else:
        print(f"No image files found in the selected class folder: {class_name}")
    
if __name__ == "__main__":
    main()