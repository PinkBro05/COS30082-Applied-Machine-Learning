import argparse
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from utils.data_collector import Datacollector
from models import ResNetCustom

def train_model(
    model_id,
    train_data,
    val_data,
    test_data,
    class_names,
    epochs=60,
    learning_rate=0.001,
    output_dir='saved_models',
):
    """
    Train a model
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        test_data: Test dataset
        class_names: List of class names
        epochs: Number of epochs to train
        learning_rate: Learning rate
        output_dir: Directory to save model and checkpoints
        use_class_weights: Whether to use class weights for balanced training
        unfreeze_layers: Number of layers to unfreeze for fine-tuning (only for ResNet50)
        
    Returns:
        Trained model and training history
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize model based on type
    num_classes = len(class_names)
    input_shape = (64, 64, 3)
    
    print(f"Creating {model_id} model with {num_classes} classes...")
    
    
    model = ResNetCustom(input_shape=input_shape, num_classes=num_classes)
    model.build()
    
    # Display model summary
    model.summary()
    
    # Define callbacks
    model_name = f"{model_id}_model"
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        
        # Save best model
        ModelCheckpoint(
            filepath=os.path.join(output_dir, f"best_{model_name}.h5"),
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        ),
        
        # Reduce learning rate when a metric has stopped improving
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adadelta(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    print(f"\nTraining {model_name} for {epochs} epochs...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
    )
    
    # Save the final model
    final_model_path = os.path.join(output_dir, f"{model_name}.h5")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Evaluate on test data
    print("\nEvaluating on test data...")
    test_loss, test_accuracy = model.evaluate(test_data)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plot_training_history(history, model_name)
    
    return model, history

def plot_training_history(history, model_name):
    """Plot and save training history"""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'saved_figures/{model_name}_training_history.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train image classification models')
    parser.add_argument('--model_id', type=str, default='test',
                        help='Model ID for saving the model')
    parser.add_argument('--epochs', type=int, default=60, 
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size')
    parser.add_argument('--train_path', type=str, default='data/classification_data/train_data', 
                        help='Path to training data directory')
    parser.add_argument('--val_path', type=str, default='data/classification_data/val_data',
                        help='Path to validation data directory')
    parser.add_argument('--test_path', type=str, default='data/classification_data/test_data', 
                        help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='model_factory', 
                        help='Directory to save model and checkpoints')
    
    args = parser.parse_args()
      # Initialize DataCollector and load data
    data = Datacollector(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path
    )
    
    # Load the classification data
    train_ds, val_ds, test_ds = data.load_data()
    
    # Get class names from the loaded data
    class_names = data.get_class_names()
    
    # Make sure we have class names before proceeding
    if not class_names:
        print("Error: Unable to get class names from the data. Please check your data directories.")
        return
    
    # Train the model
    train_model(
        model_id=args.model_id,
        train_data=train_ds,
        val_data=val_ds,
        test_data=test_ds,
        class_names=class_names,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )
    
if __name__ == "__main__":
    main()