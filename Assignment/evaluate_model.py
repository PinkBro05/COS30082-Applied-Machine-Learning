import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

from data_collector import Datacollector

def evaluate_model(model_path, test_data_path='Data/test'):
    """
    Evaluate the model with metrics:
    1. Top-1 accuracy
    2. Average Accuracy per Class
    
    Args:
        model_path: Path to the trained model
        test_data_path: Path to the test data directory
    """
    # Load the model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Initialize DataCollector and load test data
    data_collector = Datacollector(train_path="Data/train", test_path=test_data_path)
    _, _, test_data = data_collector.split_data(
        batch_size=32, 
        img_height=224, 
        img_width=224,
        use_resnet_preprocessing=True  # Use the same preprocessing as training
    )
    
    # Get class names
    class_names = data_collector.get_class_names()
    num_classes = len(class_names)
    
    # Collect all predictions and true labels
    all_predictions = []
    all_true_labels = []
    all_pred_probs = []
    
    print("\nEvaluating model performance...")
    
    # Process batches and collect predictions
    for images, labels in test_data:
        # Make predictions
        batch_predictions = model.predict(images, verbose=0)
        pred_labels = np.argmax(batch_predictions, axis=1)
        
        # Store predictions, probabilities, and true labels
        all_predictions.extend(pred_labels)
        all_true_labels.extend(labels.numpy())
        all_pred_probs.extend(batch_predictions)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    all_pred_probs = np.array(all_pred_probs)
    
    # Calculate Top-1 overall accuracy
    top1_accuracy = np.mean(all_predictions == all_true_labels)
    print(f"\nTop-1 Overall Accuracy: {top1_accuracy:.4f}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    # Calculate accuracy per class
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
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
    
    # Visualize results
    visualize_results(
        class_names=class_names,
        top1_accuracy=top1_accuracy, 
        avg_accuracy_per_class=avg_accuracy_per_class,
        per_class_accuracy=per_class_accuracy,
        confusion_matrix=cm
    )
    
    return {
        'top1_accuracy': top1_accuracy,
        'per_class_accuracy': per_class_accuracy,
        'average_accuracy_per_class': avg_accuracy_per_class,
        'confusion_matrix': cm
    }

def visualize_results(class_names, top1_accuracy, avg_accuracy_per_class, per_class_accuracy, confusion_matrix):
    """Visualize evaluation results"""
    # Create a figure with 2 subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 16))
    
    # Plot per-class accuracy
    sns.barplot(x=class_names, y=per_class_accuracy, ax=axs[0])
    axs[0].set_title(f'Accuracy by Class\nTop-1 Accuracy: {top1_accuracy:.4f}, Average Accuracy per Class: {avg_accuracy_per_class:.4f}')
    axs[0].set_xlabel('Class')
    axs[0].set_ylabel('Accuracy')
    axs[0].axhline(y=avg_accuracy_per_class, color='r', linestyle='--', label='Average Accuracy')
    axs[0].axhline(y=top1_accuracy, color='g', linestyle='--', label='Top-1 Accuracy')
    axs[0].set_xticklabels(class_names, rotation=45, ha='right')
    axs[0].legend()
    
    # Plot confusion matrix
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axs[1])
    axs[1].set_title('Confusion Matrix')
    axs[1].set_xlabel('Predicted Label')
    axs[1].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300)
    plt.show()

def main():
    # Path to the model file
    model_path = 'models/resnet_fine_tuned_final.h5'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
        available_models = [f for f in os.listdir('models/') if f.endswith('.h5')]
        
        if available_models:
            print("\nAvailable models:")
            for i, model_name in enumerate(available_models):
                print(f"{i+1}. models/{model_name}")
            
            choice = input("\nEnter the number of the model you want to evaluate: ")
            try:
                model_idx = int(choice) - 1
                if 0 <= model_idx < len(available_models):
                    model_path = os.path.join('models', available_models[model_idx])
                    print(f"Using model: {model_path}")
                else:
                    print("Invalid choice. Exiting.")
                    return
            except ValueError:
                print("Invalid input. Exiting.")
                return
        else:
            print("No models found in 'models/' directory. Exiting.")
            return
    
    # Evaluate the model
    print(f"Evaluating model: {model_path}")
    results = evaluate_model(model_path)
    
if __name__ == "__main__":
    main()