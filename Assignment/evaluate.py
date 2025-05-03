import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from ultis.data_collector import Datacollector
from models import ResNet50Model, ResNetCustom, BasicCNN

def evaluate_model(model, test_data, class_names):
    """
    Evaluate the model on test data
    
    Args:
        model: The model to evaluate
        test_data: Test dataset
        class_names: List of class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Collect all predictions and true labels
    all_predictions = []
    all_true_labels = []
    
    for images, labels in test_data:
        predictions = model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)
        
        all_predictions.extend(predicted_classes)
        all_true_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    # Calculate top-1 accuracy
    top1_accuracy = np.mean(all_predictions == all_true_labels)
    
    # Calculate per-class accuracy
    per_class_accuracy = []
    for class_idx in range(len(class_names)):
        # Get indices of samples from this class
        indices = np.where(all_true_labels == class_idx)[0]
        if len(indices) > 0:
            # Calculate accuracy for this class
            class_accuracy = np.mean(all_predictions[indices] == all_true_labels[indices])
            per_class_accuracy.append(class_accuracy)
        else:
            per_class_accuracy.append(0.0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    # Display metrics
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    
    # Display per-class accuracy
    print("\nAccuracy per class:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {per_class_accuracy[i]:.4f}")
    
    # Calculate average accuracy across all classes (balanced accuracy)
    avg_accuracy_per_class = np.mean(per_class_accuracy)
    print(f"\nAverage Accuracy per Class: {avg_accuracy_per_class:.4f}")
    
    # Calculate standard deviation of per-class accuracy
    std_accuracy = np.std(per_class_accuracy)
    print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_predictions, target_names=class_names))
    
    # Visualize results
    visualize_results(
        class_names=class_names,
        top1_accuracy=top1_accuracy, 
        avg_accuracy_per_class=avg_accuracy_per_class,
        per_class_accuracy=per_class_accuracy,
        confusion_matrix=cm
    )
    
    return {
        "top1_accuracy": top1_accuracy,
        "avg_accuracy_per_class": avg_accuracy_per_class,
        "std_accuracy": std_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": cm
    }

def visualize_results(class_names, top1_accuracy, avg_accuracy_per_class, per_class_accuracy, confusion_matrix):
    """
    Visualize evaluation results
    
    Args:
        class_names: List of class names
        top1_accuracy: Top-1 accuracy
        avg_accuracy_per_class: Average accuracy per class
        per_class_accuracy: List of per-class accuracies
        confusion_matrix: Confusion matrix
    """
    # Create figure with 2 subplots
    plt.figure(figsize=(18, 8))
    
    # Plot per-class accuracy
    plt.subplot(1, 2, 1)
    plt.bar(class_names, per_class_accuracy)
    plt.axhline(y=avg_accuracy_per_class, color='r', linestyle='-', label=f'Average: {avg_accuracy_per_class:.4f}')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title(f'Per-Class Accuracy (Overall: {top1_accuracy:.4f})')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    
    # Plot confusion matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('saved_figures/evaluation_results.png')
    plt.show()

def analyze_class_bias(model, test_data_path='Data/test'):
    """
    Analyze if there's a bias in model predictions related to class order
    
    Args:
        model: The model to analyze
        test_data_path: Path to test data directory
    """
    # Get class names in the same order as used during training
    class_names = get_class_names_from_directory("Data/train")
    print(f"Classes (in alphabetical order): {class_names}")
    
    # Store all predictions
    all_predictions = []
    true_classes = []
    
    # Process each class
    for class_idx, class_name in enumerate(class_names):
        # Path to class folder in test data
        class_folder = os.path.join(test_data_path, class_name)
        
        # Get all image files
        image_files = [os.path.join(class_folder, f) for f in os.listdir(class_folder)
                      if os.path.isfile(os.path.join(class_folder, f)) and
                      f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit number of images per class (for faster analysis)
        max_images = 30
        if len(image_files) > max_images:
            image_files = image_files[:max_images]
        
        # Process each image
        for image_path in image_files:
            # Load and preprocess image
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = img_array / 255.0  # Normalize
            
            # Get prediction
            predictions = model.predict(img_array)
            
            # Store prediction and true class
            all_predictions.append(predictions[0])
            true_classes.append(class_idx)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    true_classes = np.array(true_classes)
    
    # Calculate average prediction probability for each class
    avg_predictions = np.zeros((len(class_names), len(class_names)))
    
    for true_idx in range(len(class_names)):
        # Get predictions for this class
        class_predictions = all_predictions[true_classes == true_idx]
        
        if len(class_predictions) > 0:
            # Calculate average prediction for each output node
            avg_predictions[true_idx] = np.mean(class_predictions, axis=0)
    
    # Convert to pandas DataFrame for better visualization
    avg_df = pd.DataFrame(avg_predictions, 
                         index=class_names,
                         columns=class_names)
    
    print("\nAverage prediction probabilities for each class:")
    print(avg_df)
    
    # Visualize average prediction distribution
    plt.figure(figsize=(12, 10))
    sns.heatmap(avg_df, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Average Prediction Probabilities by Class")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig('saved_figures/class_bias_heatmap.png')
    
    # Calculate average prediction by position (regardless of class)
    avg_by_position = np.mean(all_predictions, axis=0)
    
    # Plot average prediction by position
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, avg_by_position)
    plt.xlabel('Class (in alphabetical order)')
    plt.ylabel('Average Prediction Probability')
    plt.title('Average Prediction Probability by Class Position')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('saved_figures/avg_prediction_by_position.png')
    
    # Check if there's a correlation between position and average prediction
    position = np.arange(len(class_names))
    correlation = np.corrcoef(position, avg_by_position)[0, 1]
    print(f"\nCorrelation between class position and average prediction: {correlation:.4f}")
    
    if abs(correlation) > 0.3:
        if correlation < 0:
            print("There appears to be a negative correlation between class position and prediction probability.")
            print("This suggests your model may have a bias favoring classes that come earlier alphabetically.")
        else:
            print("There appears to be a positive correlation between class position and prediction probability.")
            print("This suggests your model may have a bias favoring classes that come later alphabetically.")
    else:
        print("There doesn't appear to be a strong correlation between class position and prediction.")
        print("The bias you're observing might be due to other factors in your training data or model.")
    
    # Check class imbalance in test data
    counts = []
    for class_name in class_names:
        class_folder = os.path.join(test_data_path, class_name)
        count = len([f for f in os.listdir(class_folder) 
                     if os.path.isfile(os.path.join(class_folder, f))])
        counts.append(count)
    
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, counts)
    plt.xlabel('Class')
    plt.ylabel('Number of Test Images')
    plt.title('Test Data Class Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('saved_figures/test_class_distribution.png')
    plt.show()

def get_class_names_from_directory(data_path="Data/train"):
    """Get class names in alphabetical order as used during training"""
    class_names = sorted([item.name for item in os.scandir(data_path) 
                         if item.is_dir() and not item.name.startswith('.')])
    return class_names

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model file')
    parser.add_argument('--model_type', type=str, default='resnet50', 
                        choices=['resnet50', 'resnet_custom', 'basic_cnn', 'generic'],
                        help='Type of model to evaluate')
    parser.add_argument('--test_data_path', type=str, default='Data/test', 
                        help='Path to test data directory')
    parser.add_argument('--analyze_bias', action='store_true', 
                        help='Analyze potential class bias in the model')
    
    args = parser.parse_args()
    
    # Get class names
    class_names = get_class_names_from_directory("Data/train")
    num_classes = len(class_names)
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    if args.model_type == 'resnet50':
        model = ResNet50Model.load(args.model_path)
    elif args.model_type == 'resnet_custom':
        model = ResNetCustom.load(args.model_path)
    elif args.model_type == 'basic_cnn':
        model = BasicCNN.load(args.model_path)
    else:
        # Generic loading
        model = tf.keras.models.load_model(args.model_path)
    
    # If analyzing bias, run that analysis
    if args.analyze_bias:
        print("\nAnalyzing potential class bias in the model...")
        analyze_class_bias(model, args.test_data_path)
        return

    # Initialize DataCollector and load the test data
    data_collector = Datacollector(train_path="Data/train", test_path=args.test_data_path)
    _, _, test_data = data_collector.split_data(
        batch_size=32, 
        img_height=224, 
        img_width=224,
        use_resnet_preprocessing=True if args.model_type == 'resnet50' else False
    )
    
    # Evaluate the model
    print("\nEvaluating model performance...")
    evaluate_model(model, test_data, class_names)
    
if __name__ == "__main__":
    main()