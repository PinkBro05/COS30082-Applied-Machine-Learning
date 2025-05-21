import argparse
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from utils.data_collector import Datacollector
from models import ResNetCustom

def load_model(model_path):
    """
    Load a trained model from path
    
    Args:
        model_path: Path to the saved model (.h5 file)
        
    Returns:
        Loaded model
    """
    print(f"Loading model from: {model_path}")
    try:
        # Import the custom layers needed
        from models.resnet_custom import ResidualBlock, ArcFaceHead
        
        # Define the custom objects dictionary
        custom_objects = {
            'ResidualBlock': ResidualBlock,
            'ArcFaceHead': ArcFaceHead
        }
        
        # Load the model with custom objects
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def evaluate_model(model, test_data, class_names):
    """
    Evaluate the model on test data
    
    Args:
        model: Loaded model
        test_data: Test dataset
        class_names: List of class names
        
    Returns:
        Evaluation metrics
    """
    print("\nEvaluating model on test data...")
    
    # Extract the test images and labels from the dataset
    all_images = []
    all_labels = []
    
    # Handle different dataset formats
    for batch in test_data:
        # Check if the batch is a tuple or a dictionary (ArcFace format)
        if isinstance(batch, tuple):
            images, labels = batch
            if isinstance(images, dict) and 'input_1' in images:
                # ArcFace format with dict
                all_images.extend(images['input_1'].numpy())
                all_labels.extend(labels.numpy())
            else:
                # Standard format
                all_images.extend(images.numpy())
                all_labels.extend(labels.numpy())
        else:
            raise ValueError("Unexpected dataset format")
    
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
      # Evaluate the model
    print("Running predictions...")
    try:
        # First try using the dataset directly (preferred approach)
        predictions = model.predict(test_data)
    except Exception as e:
        print(f"Error with dataset prediction: {e}")
        print("Trying alternative prediction approach...")
        
        # Check if model has multiple inputs (like ArcFace models)
        if isinstance(model.input, list) or (hasattr(model.input, '_keras_shape') and len(model.input._keras_shape) > 1):
            # Create dummy labels for ArcFace
            dummy_labels = np.zeros(len(all_images), dtype=np.int32)
            try:
                # Try prediction with dictionary format
                predictions = model.predict({'input_1': all_images, 'label_input': dummy_labels})
            except Exception:
                # Try with list format
                predictions = model.predict([all_images, dummy_labels])
        else:
            # Standard prediction for regular models
            predictions = model.predict(all_images)
    
    # Get predicted classes
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Print evaluation metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, predicted_classes, target_names=class_names))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, predicted_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('saved_figures/confusion_matrix.png')
    plt.show()
    
    # Calculate overall accuracy
    accuracy = np.mean(predicted_classes == all_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    return accuracy, predictions

def visualize_predictions(test_data, predictions, class_names, num_samples=16):
    """
    Visualize some predictions from the test dataset
    
    Args:
        test_data: Test dataset
        predictions: Model predictions
        class_names: List of class names
        num_samples: Number of samples to visualize
    """
    # Extract images and labels
    images = []
    labels = []
    
    # Get samples from the test dataset
    for batch in test_data.take(1):  # Take just one batch
        if isinstance(batch, tuple):
            if isinstance(batch[0], dict) and 'input_1' in batch[0]:
                # ArcFace format
                images = batch[0]['input_1'].numpy()
                labels = batch[1].numpy()
            else:
                # Standard format
                images = batch[0].numpy()
                labels = batch[1].numpy()
    
    # Ensure we don't try to show more images than we have
    num_samples = min(num_samples, len(images))
    
    # Get predictions for these samples
    preds = np.argmax(predictions[:len(images)], axis=1)
    
    # Visualize the predictions
    plt.figure(figsize=(15, 12))
    for i in range(num_samples):
        plt.subplot(4, 4, i+1)
        
        # For ResNet preprocess_input, we need to convert back for display
        # ResNet preprocessing shifts the pixel values, adding the ImageNet mean back
        img = images[i].copy()
        
        # If using ResNet preprocessing (values outside 0-255 range)
        if np.min(img) < 0 or np.max(img) > 255:
            # Convert from ResNet preprocessing back to displayable image
            # These values reverse the effects of the preprocess_input function
            img = img + np.array([123.68, 116.779, 103.939])
            img = img[...,::-1]  # BGR to RGB
            img = np.clip(img, 0, 255).astype('uint8')
        else:
            # Standard format, just convert to uint8
            img = img.astype('uint8')
        
        plt.imshow(img)
        
        # Color green for correct predictions, red for incorrect
        color = 'green' if preds[i] == labels[i] else 'red'
        title = f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}"
        
        plt.title(title, color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('saved_figures/prediction_samples.png')
    plt.show()

def inference_single_samples(model, test_path, class_names, num_samples=3, use_arcface=False):
    """
    Perform inference on individual samples from the test folder
    
    Args:
        model: Loaded model
        test_path: Path to test data directory
        class_names: List of class names
        num_samples: Number of random samples to predict
        use_arcface: Whether the model uses ArcFace format
    """
    
    print(f"\nPerforming inference on {num_samples} random samples...")
    
    # Get all class subdirectories
    class_dirs = [os.path.join(test_path, class_name) for class_name in class_names]
    
    # Get sample images from random classes
    sample_images = []
    true_labels = []
    file_paths = []
    
    # Select random samples across different classes
    for _ in range(num_samples):
        # Pick a random class directory
        class_dir = random.choice(class_dirs)
        class_name = os.path.basename(class_dir)
        class_idx = class_names.index(class_name)
        
        # List all image files in this class directory
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            # Select a random image file
            img_file = random.choice(image_files)
            img_path = os.path.join(class_dir, img_file)
            file_paths.append(img_path)
            
            # Load and preprocess the image
            img = image.load_img(img_path, target_size=(64, 64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            processed_img = preprocess_input(img_array)
            
            sample_images.append(processed_img)
            true_labels.append(class_idx)
    
    # Concatenate all images
    if sample_images:
        batch_images = np.vstack(sample_images)
        
        # Make predictions
        try:
            if use_arcface:
                # For ArcFace models
                dummy_labels = np.zeros(len(sample_images), dtype=np.int32)
                try:
                    # Try prediction with dictionary format
                    predictions = model.predict({'input_1': batch_images, 'label_input': dummy_labels})
                except Exception:
                    # Try with list format
                    predictions = model.predict([batch_images, dummy_labels])
            else:
                # Standard prediction
                predictions = model.predict(batch_images)
                
            # Get predicted classes
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Visualize predictions
            plt.figure(figsize=(15, 5 * num_samples))
            for i in range(len(sample_images)):
                plt.subplot(num_samples, 1, i + 1)
                
                # Load original image for display
                img = image.load_img(file_paths[i], target_size=(64, 64))
                plt.imshow(img)
                
                # Color green for correct predictions, red for incorrect
                color = 'green' if predicted_classes[i] == true_labels[i] else 'red'
                confidence = predictions[i][predicted_classes[i]] * 100
                
                title = f"File: {os.path.basename(file_paths[i])}\n"
                title += f"True: {class_names[true_labels[i]]}\n" 
                title += f"Pred: {class_names[predicted_classes[i]]} ({confidence:.2f}%)"
                
                plt.title(title, color=color, fontsize=12)
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig('saved_figures/inference_samples.png')
            plt.show()
            
            print("\nInference results:")
            for i in range(len(sample_images)):
                print(f"Sample {i+1}:")
                print(f"  File: {file_paths[i]}")
                print(f"  True class: {class_names[true_labels[i]]}")
                print(f"  Predicted class: {class_names[predicted_classes[i]]}")
                print(f"  Confidence: {predictions[i][predicted_classes[i]]:.4f}")
                print()
                
        except Exception as e:
            print(f"Error during inference: {e}")
    else:
        print("No sample images found in the test directory.")

def main():
    parser = argparse.ArgumentParser(description='Inference for image classification models')
    parser.add_argument('--mode', type=int, default=0, choices=[0, 1],
                        help='0: Testing mode (evaluate on test set), 1: Inference mode (predict individual samples)')
    parser.add_argument('--model_path', type=str, default='model_factory/best_test_model.h5',
                        help='Path to trained model file (.h5)')
    parser.add_argument('--test_path', type=str, default='data/classification_data/test_data',
                        help='Path to test data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--use_arcface', action='store_true',
                        help='Enable if the model was trained with ArcFace')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples for inference mode (mode 1)')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.isfile(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    # Create directory for saved figures if it doesn't exist
    if not os.path.exists('saved_figures'):
        os.makedirs('saved_figures')
      # Initialize DataCollector and load data
    data = Datacollector(test_path=args.test_path)
    
    # Always load at least a small batch of data to get class names
    _, _, test_ds = data.load_data(
        batch_size=args.batch_size,
        use_arcface=args.use_arcface
    )
    
    # Get class names
    class_names = data.get_class_names()
    if not class_names:
        print("Error: Unable to get class names from the data.")
        return
    
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Load the model
    model = load_model(args.model_path)
    if model is None:
        return
    
    # Print model summary
    model.summary()
    
    # Run in the selected mode
    if args.mode == 0:
        # Testing mode: Evaluate on test dataset
        print("\nRunning in Testing Mode (evaluate on test set)")
        accuracy, predictions = evaluate_model(model, test_ds, class_names)
        print(f"\nTesting completed. Results saved to 'saved_figures/' directory.")
    else:
        # Inference mode: Predict individual samples
        print("\nRunning in Inference Mode (predict individual samples)")
        inference_single_samples(
            model=model, 
            test_path=args.test_path, 
            class_names=class_names, 
            num_samples=args.num_samples,
            use_arcface=args.use_arcface
        )
        print(f"\nInference completed. Results saved to 'saved_figures/' directory.")

if __name__ == "__main__":
    main()
