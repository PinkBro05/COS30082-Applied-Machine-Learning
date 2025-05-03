import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import random
import os

from models import ResNet50Model, ResNetCustom, BasicCNN

def predict_single_image(model, image_path, class_names, model_type='resnet50'):
    """
    Make a prediction on a single image
    
    Args:
        model: Loaded Keras model
        image_path: Path to the image file
        class_names: List of class names
        model_type: Type of model ('resnet50', 'resnet_custom', or 'basic_cnn')
    """
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Apply appropriate preprocessing based on model type
    if model_type in ['resnet50', 'resnet_custom']:
        # Use ResNet-specific preprocessing
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        print("Using ResNet preprocessing")
    else:
        # Standard normalization for other models
        img_array = img_array / 255.0
        print("Using standard normalization (/255)")
    
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

def load_model_by_type(model_type, model_path=None, input_shape=(224, 224, 3), num_classes=10):
    """
    Load a model either from a saved file or create a new one
    
    Args:
        model_type: Type of model to load ('resnet50', 'resnet_custom', or 'basic_cnn')
        model_path: Path to saved model file (if None, creates a new model)
        input_shape: Shape of input images
        num_classes: Number of classes for classification
        
    Returns:
        Loaded model
    """
    if model_path:
        print(f"Loading model from {model_path}")
        if model_type == 'resnet50':
            return ResNet50Model.load(model_path)
        elif model_type == 'resnet_custom':
            return ResNetCustom.load(model_path)
        elif model_type == 'basic_cnn':
            return BasicCNN.load(model_path)
        else:
            # Generic model loading
            return tf.keras.models.load_model(model_path)
    else:
        print(f"Creating new {model_type} model")
        if model_type == 'resnet50':
            model = ResNet50Model(input_shape=input_shape, num_classes=num_classes)
            model.build()
            return model
        elif model_type == 'resnet_custom':
            model = ResNetCustom(input_shape=input_shape, num_classes=num_classes)
            model.build()
            return model
        elif model_type == 'basic_cnn':
            model = BasicCNN(input_shape=input_shape, num_classes=num_classes)
            model.build()
            return model
        else:
            raise ValueError(f"Unknown model type: {model_type}")

def main():
    parser = argparse.ArgumentParser(description='Predict images using a pre-trained model')
    parser.add_argument('--model_path', type=str, help='Path to the saved model file')
    parser.add_argument('--model_type', type=str, default='resnet50', 
                        choices=['resnet50', 'resnet_custom', 'basic_cnn'],
                        help='Type of model to use (if no model_path is provided)')
    parser.add_argument('--image_path', type=str, help='Path to specific image to predict (optional)')
    parser.add_argument('--data_path', type=str, default='Data/test', 
                        help='Path to test data directory (used if image_path is not provided)')
    
    args = parser.parse_args()
    
    # Get class names in the same order as used during training
    class_names = get_class_names_from_directory("Data/train")
    num_classes = len(class_names)
    print(f"Using class names in alphabetical order: {class_names}")
    
    # Load model
    model = load_model_by_type(
        model_type=args.model_type,
        model_path=args.model_path,
        input_shape=(224, 224, 3),
        num_classes=num_classes
    )
    
    # If a specific image path is provided, use that
    if args.image_path:
        image_path = args.image_path
        # Try to determine the actual class from the directory structure
        try:
            class_name = os.path.basename(os.path.dirname(image_path))
            print(f"Actual class (from path): {class_name}")
        except:
            class_name = "Unknown"
    else:
        # Choose a random test image to predict
        test_dir = pathlib.Path(args.data_path)
        
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
        else:
            print(f"No image files found in the selected class folder: {class_name}")
            return
    
    print(f"\nSelected image: {image_path}")
    print(f"Actual class: {class_name}")
    
    # Make prediction
    result = predict_single_image(model, image_path, class_names, model_type=args.model_type)
    
    # Print results
    print(f"Predicted class: {result['class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nTop 3 predictions:")
    top_predictions = sorted(result['predictions'].items(), key=lambda x: x[1], reverse=True)[:3]
    for class_name, confidence in top_predictions:
        print(f"- {class_name}: {confidence:.2%}")
    
    # Visualize the prediction
    visualize_prediction(image_path, result)
    
if __name__ == "__main__":
    main()