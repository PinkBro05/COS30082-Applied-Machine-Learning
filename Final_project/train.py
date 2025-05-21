import argparse
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

from utils.data_collector import Datacollector
from models import ResNetCustom

class WarmupCosineDecayScheduler(Callback):
    """
    Cosine decay learning rate scheduler with linear warmup.
    
    This callback gradually increases the learning rate from 0 to the base learning rate
    over the warmup period, then applies cosine decay from the base learning rate to 0
    over the remaining epochs.
    
    Args:
        base_lr: Base learning rate after warmup.
        total_epochs: Total number of training epochs.
        warmup_epochs: Number of warmup epochs (default=1).
        verbose: Whether to print learning rate updates (0=silent, 1=update messages).
    """
    def __init__(self, base_lr, total_epochs, warmup_epochs=1, verbose=1):
        super(WarmupCosineDecayScheduler, self).__init__()
        self.base_lr = base_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose
        self.learning_rates = []
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear warmup phase
            lr = self.base_lr * ((epoch + 1) / self.warmup_epochs)
        else:
            # Cosine decay phase with minimum learning rate
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            # Ensure the learning rate decays all the way to a very small value by the last epoch
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            
            # Add a small minimum value to avoid exactly zero
            min_lr = 1e-6  # Small minimum learning rate
            lr = max(lr, min_lr)
        
        # Set the learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.learning_rates.append(lr)
        
        if self.verbose > 0:
            print(f"\nEpoch {epoch+1}: Learning rate set to {lr:.6f}")
            
    def plot_lr_schedule(self):
        """Plot the learning rate schedule"""
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(self.learning_rates) + 1), self.learning_rates, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.show()

def train_model(
    model_id,
    is_arcface,
    train_data,
    val_data,
    test_data,
    class_names,
    epochs=30,
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
    model.build(is_acrf=is_arcface)
    
    # Display model summary
    model.summary()
      # Define callbacks
    model_name = f"{model_id}_model"
    
    # Create the learning rate scheduler with warmup and cosine decay
    lr_scheduler = WarmupCosineDecayScheduler(
        base_lr=learning_rate,
        total_epochs=epochs,
        warmup_epochs=1,  # 1 epoch of warmup
        verbose=1
    )
    
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
        
        # Add our custom LR scheduler
        lr_scheduler
    ]
    
    # Compile the model with a fixed initial learning rate
    # The scheduler will handle dynamic learning rate changes
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.0,  # Initial LR will be set by scheduler
                                          momentum=0.9,
                                          decay=5e-4,
                                          ),
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
    
    # Plot training history and learning rate schedule
    plot_training_history(history, model_name)
    
    # Plot the learning rate schedule
    for callback in callbacks:
        if isinstance(callback, WarmupCosineDecayScheduler):
            plt.figure(figsize=(10, 4))
            plt.plot(range(1, len(callback.learning_rates) + 1), callback.learning_rates, marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.grid(True)
            plt.savefig(f'saved_figures/{model_name}_lr_schedule.png')
            plt.show()
            break
    
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
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512, 
                        help='Batch size')
    parser.add_argument('--use_arcface', default=False, type=bool,
                        help='Use ArcFace format for data')
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
    
    # Load the classification data with ArcFace format enabled
    train_ds, val_ds, test_ds = data.load_data(
        batch_size=args.batch_size,
        use_arcface=args.use_arcface  # Enable ArcFace format
    )
    
    # Get class names from the loaded data
    class_names = data.get_class_names()
    
    # Make sure we have class names before proceeding
    if not class_names:
        print("Error: Unable to get class names from the data. Please check your data directories.")
        return
    
    # Train the model
    train_model(
        is_arcface=args.use_arcface,
        model_id=args.model_id,
        train_data=train_ds,
        val_data=val_ds,
        test_data=test_ds,
        class_names=class_names,
        epochs=args.epochs,
        # learning_rate=args.learning_rate,
        learning_rate= 0.05*(args.batch_size/512),
        output_dir=args.output_dir,
    )
    
if __name__ == "__main__":
    main()