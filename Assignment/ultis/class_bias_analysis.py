import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from tqdm import tqdm
import pandas as pd
import seaborn as sns

def get_class_names_from_directory(data_path="Data/train"):
    """Get class names in alphabetical order as used during training"""
    class_names = sorted([item.name for item in os.scandir(data_path) 
                         if item.is_dir() and not item.name.startswith('.')])
    return class_names

def analyze_class_bias(model_path='models/best_resnet_model_balanced.h5', test_data_path='Data/test'):
    """
    Analyze if there's a bias in model predictions related to class order
    """
    # Load the model
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # Get class names in the same order as used during training
    class_names = get_class_names_from_directory("Data/train")
    print(f"Classes (in alphabetical order): {class_names}")
    
    # Store all predictions
    all_predictions = []
    true_classes = []
    
    # Process each class
    for class_idx, class_name in enumerate(tqdm(class_names, desc="Processing classes")):
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
            predictions = model.predict(img_array, verbose=0)
            
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
    sns.heatmap(avg_df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Average Prediction Probabilities by Class")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig('class_bias_heatmap.png')
    
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
    plt.savefig('avg_prediction_by_position.png')
    
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
    plt.savefig('test_class_distribution.png')

if __name__ == "__main__":
    analyze_class_bias()