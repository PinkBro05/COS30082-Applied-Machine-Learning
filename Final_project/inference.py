import argparse
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import random

from utils.data_collector import Datacollector
from models import ResNetCustom

def load_model(model_path, model_type='supervised'):
    """
    Load a trained ResNet model from path
    
    Args:
        model_path: Path to the saved model (.h5 file)
        model_type: Type of model ('supervised', 'metric_learning', 'inference')
        
    Returns:
        Loaded ResNetCustom instance
    """
    print(f"Loading model from: {model_path}")
    try:
        # Import the custom layers needed
        from models.resnet_custom import ResidualBlock, ArcFaceHead, TripletLossLayer
        
        # Define the custom objects dictionary
        custom_objects = {
            'ResidualBlock': ResidualBlock,
            'ArcFaceHead': ArcFaceHead,
            'TripletLossLayer': TripletLossLayer
        }
        
        # Load the model with custom objects
        keras_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("Model loaded successfully")
        
        # Create a ResNetCustom wrapper
        resnet_model = ResNetCustom()
        resnet_model.model = keras_model
        resnet_model.is_inference_model = 'Inference' in keras_model.name
        
        if 'Triplet' in keras_model.name or 'Siamese' in keras_model.name:
            resnet_model.training_mode = 'metric_learning'
        else:
            resnet_model.training_mode = 'supervised'
        
        return resnet_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def evaluate_model(model, test_data, class_names):
    """
    Evaluate the ResNet model on test data
    
    Args:
        model: Loaded ResNetCustom model
        test_data: Test dataset
        class_names: List of class names
        
    Returns:
        Evaluation metrics
    """
    print("\nEvaluating model on test data...")
    
    if model.training_mode == 'metric_learning':
        print("For metric learning models, use face verification evaluation instead.")
        return None
    
    # Extract the test images and labels from the dataset
    all_images = []
    all_labels = []
    
    # Handle different dataset formats
    for batch in test_data:
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
    
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    
    # Evaluate the model
    print("Running predictions...")
    try:
        # First try using the dataset directly
        predictions = model.model.predict(test_data)
    except Exception as e:
        print(f"Error with dataset prediction: {e}")
        print("Trying alternative prediction approach...")
        
        # Check if model has multiple inputs (like ArcFace models)
        if 'ArcFace' in model.model.name:
            # Create dummy labels for ArcFace
            dummy_labels = np.zeros(len(all_images), dtype=np.int32)
            try:
                predictions = model.model.predict([all_images, dummy_labels])
            except Exception:
                predictions = model.model.predict({'input_1': all_images, 'label_input': dummy_labels})
        else:
            # Standard prediction for regular models
            predictions = model.model.predict(all_images)
    
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
        
        # Normalize image for display
        img = images[i].copy()
        
        # Normalize to 0-255 range for display
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = np.clip(img * 255, 0, 255).astype('uint8')
        else:
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

def inference_single_samples(model, test_path, class_names, num_samples=3):
    """
    Perform inference on individual samples from the test folder
    
    Args:
        model: Loaded ResNetCustom model
        test_path: Path to test data directory
        class_names: List of class names
        num_samples: Number of random samples to predict
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
            img = tf.keras.utils.load_img(img_path, target_size=(64, 64))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            # Normalize to [0, 1] range
            processed_img = img_array / 255.0
            
            sample_images.append(processed_img)
            true_labels.append(class_idx)
    
    # Concatenate all images
    if sample_images:
        batch_images = np.vstack(sample_images)
        
        # Make predictions
        try:
            if 'ArcFace' in model.model.name:
                # For ArcFace models
                dummy_labels = np.zeros(len(sample_images), dtype=np.int32)
                predictions = model.model.predict([batch_images, dummy_labels])
            else:
                # Standard prediction
                predictions = model.model.predict(batch_images)
                
            # Get predicted classes
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Visualize predictions
            plt.figure(figsize=(15, 5 * num_samples))
            for i in range(len(sample_images)):
                plt.subplot(num_samples, 1, i + 1)
                
                # Load original image for display
                img = tf.keras.utils.load_img(file_paths[i], target_size=(64, 64))
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

def evaluate_face_verification(model, verification_pairs_file, test_data_path, plot_roc=True):
    """
    Evaluate face verification performance using verification pairs
    
    Args:
        model: Loaded ResNetCustom model (should be inference model)
        verification_pairs_file: Path to verification pairs file
        test_data_path: Path to test data directory
        plot_roc: Whether to plot ROC curve
    """
    if not model.is_inference_model:
        print("Converting to inference model for face verification...")
        # Create inference version of the model
        input_shape = model.model.input_shape[1:]  # Remove batch dimension
        inference_model = ResNetCustom(input_shape=input_shape, num_classes=model.num_classes)
        inference_model.build(is_inference=True, training_mode=model.training_mode)
        
        # Copy weights from training model to inference model
        # This is a simplified approach - in practice you'd save/load the inference model
        print("Note: For proper face verification, save and load an inference model")
        return
    
    print("Loading verification pairs...")
    verification_pairs = []
    labels = []
    
    # Read verification pairs file
    with open(verification_pairs_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:  # Same person
                person_id, img1, img2 = parts
                img1_path = os.path.join(test_data_path, person_id, img1)
                img2_path = os.path.join(test_data_path, person_id, img2)
                verification_pairs.append((img1_path, img2_path))
                labels.append(1)  # Same person
            elif len(parts) == 4:  # Different persons
                person1, img1, person2, img2 = parts
                img1_path = os.path.join(test_data_path, person1, img1)
                img2_path = os.path.join(test_data_path, person2, img2)
                verification_pairs.append((img1_path, img2_path))
                labels.append(0)  # Different persons
    
    print(f"Loaded {len(verification_pairs)} verification pairs")
    
    # Convert image paths to image arrays
    image_pairs = []
    for img1_path, img2_path in verification_pairs:
        try:
            # Load and preprocess images
            img1 = tf.keras.utils.load_img(img1_path, target_size=(64, 64))
            img1_array = tf.keras.utils.img_to_array(img1) / 255.0
            
            img2 = tf.keras.utils.load_img(img2_path, target_size=(64, 64))
            img2_array = tf.keras.utils.img_to_array(img2) / 255.0
            
            image_pairs.append((img1_array, img2_array))
        except Exception as e:
            print(f"Error loading images {img1_path}, {img2_path}: {e}")
            continue
    
    # Evaluate face verification
    results = model.evaluate_face_verification(image_pairs, labels, plot_roc=plot_roc)
    return results

def main():
    parser = argparse.ArgumentParser(description='Inference for ResNet models with multiple paradigms')
    parser.add_argument('--mode', type=str, default='test', 
                        choices=['test', 'inference', 'face_verification'],
                        help='Mode: test (evaluate on test set), inference (predict samples), face_verification (ROC analysis)')
    parser.add_argument('--model_path', type=str, default='model_factory/best_resnet_custom_supervised_model.h5',
                        help='Path to trained model file (.h5)')
    parser.add_argument('--test_path', type=str, default='data/classification_data/test_data',
                        help='Path to test data directory')
    parser.add_argument('--verification_path', type=str, default='data/verification_data',
                        help='Path to verification data directory (for face verification mode)')
    parser.add_argument('--verification_pairs', type=str, default='data/verification_pairs_test.txt',
                        help='Path to verification pairs file (for face verification mode)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples for inference mode')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.isfile(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    # Create directory for saved figures if it doesn't exist
    if not os.path.exists('saved_figures'):
        os.makedirs('saved_figures')
    
    print(f"Running in {args.mode} mode")
    
    # Load the model
    model = load_model(args.model_path)
    if model is None:
        return
    
    print(f"Model type: {model.training_mode}")
    print(f"Is inference model: {model.is_inference_model}")
    
    # Print model summary
    model.model.summary()
    
    if args.mode == 'face_verification':
        # Face verification mode (for metric learning models)
        if not model.is_inference_model:
            print("Warning: Face verification works best with inference models")
        
        evaluate_face_verification(
            model, 
            args.verification_pairs, 
            args.verification_path,
            plot_roc=True
        )
        
    elif args.mode == 'test':
        # Testing mode: Evaluate on test dataset (for supervised models)
        if model.training_mode != 'supervised':
            print("Test mode is only available for supervised models")
            return
            
        # Initialize DataCollector and load data
        data_collector = Datacollector(test_path=args.test_path)
        
        # Determine if model uses ArcFace
        use_arcface = 'ArcFace' in model.model.name
        
        _, _, test_ds = data_collector.load_data(
            batch_size=args.batch_size,
            use_arcface=use_arcface
        )
        
        # Get class names
        class_names = data_collector.get_class_names()
        if not class_names:
            print("Error: Unable to get class names from the data.")
            return
        
        print(f"Found {len(class_names)} classes: {class_names}")
        
        # Evaluate model
        accuracy, predictions = evaluate_model(model, test_ds, class_names)
        
        # Visualize some predictions
        visualize_predictions(test_ds, predictions, class_names)
        
        print(f"\nTesting completed. Results saved to 'saved_figures/' directory.")
        
    else:  # inference mode
        # Inference mode: Predict individual samples
        if model.training_mode != 'supervised':
            print("Inference mode is only available for supervised models")
            return
            
        # Get class names from directory structure
        class_names = [d for d in os.listdir(args.test_path) 
                      if os.path.isdir(os.path.join(args.test_path, d))]
        class_names.sort()
        
        if not class_names:
            print("Error: No class directories found in test path")
            return
            
        print(f"Found {len(class_names)} classes: {class_names}")
        
        inference_single_samples(model, args.test_path, class_names, args.num_samples)

if __name__ == "__main__":
    main()
