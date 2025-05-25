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
    training_mode='supervised',
    use_arcface=False,
    train_data=None,
    val_data=None,
    test_data=None,
    class_names=None,
    epochs=30,
    learning_rate=0.001,
    output_dir='model_factory',
    batch_size=32,
    images=None,
    labels=None
):
    """
    Train a ResNet model with support for both supervised and metric learning
    
    Args:
        model_id: Identifier for the model
        training_mode: 'supervised' or 'metric_learning'
        use_arcface: Whether to use ArcFace head (only for supervised mode)
        train_data: Training dataset (for supervised mode)
        val_data: Validation dataset (for supervised mode)
        test_data: Test dataset (for supervised mode)
        class_names: List of class names
        epochs: Number of epochs to train
        learning_rate: Learning rate
        output_dir: Directory to save model and checkpoints
        batch_size: Batch size (for metric learning)
        images: Image array (for metric learning)
        labels: Label array (for metric learning)
        
    Returns:
        Trained model and training history
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists('saved_figures'):
        os.makedirs('saved_figures')
    
    # Initialize model
    num_classes = len(class_names) if class_names else 10
    input_shape = (64, 64, 3)
    
    print(f"Creating {model_id} model with {num_classes} classes...")
    print(f"Training mode: {training_mode}")
    if training_mode == 'supervised':
        print(f"Using ArcFace: {use_arcface}")
    
    # Create and build model
    model = ResNetCustom(input_shape=input_shape, num_classes=num_classes)
    model.build(is_inference=False, training_mode=training_mode, use_arcface=use_arcface)
    
    # Display model summary
    model.model.summary()
    
    # Define callbacks
    model_name = f"{model_id}_{training_mode}_model"
    if training_mode == 'supervised' and use_arcface:
        model_name += "_arcface"
    
    # Create the learning rate scheduler with warmup and cosine decay
    lr_scheduler = WarmupCosineDecayScheduler(
        base_lr=learning_rate,
        total_epochs=epochs,
        warmup_epochs=1,
        verbose=1
    )
    
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss' if training_mode == 'supervised' else 'loss',
            patience=15,
            verbose=1,
            restore_best_weights=True
        ),
        
        # Save best model
        ModelCheckpoint(
            filepath=os.path.join(output_dir, f"best_{model_name}.h5"),
            monitor='val_accuracy' if training_mode == 'supervised' else 'loss',
            verbose=1,
            save_best_only=True,
            mode='max' if training_mode == 'supervised' else 'min'
        ),
        
        # Add our custom LR scheduler
        lr_scheduler
    ]
    
    # Compile the model
    if training_mode == 'supervised':
        if use_arcface:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0)  # LR will be set by scheduler
            model.model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0)
            model.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
    else:  # metric_learning
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0)
        model.model.compile(optimizer=optimizer)
    
    # Prepare training data based on mode
    if training_mode == 'supervised':
        if train_data is None or val_data is None:
            raise ValueError("For supervised learning, train_data and val_data must be provided")
        
        print(f"\nTraining {model_name} for {epochs} epochs...")
        history = model.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
        )
        
        # Evaluate on test data if provided
        if test_data is not None:
            print("\nEvaluating on test data...")
            test_loss, test_accuracy = model.model.evaluate(test_data)
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    else:  # metric_learning
        if images is None or labels is None:
            raise ValueError("For metric learning, images and labels arrays must be provided")
        
        # Create triplet dataset
        print("Creating triplet dataset for metric learning...")
        triplet_dataset = model.create_triplet_dataset(images, labels, batch_size=batch_size)
        
        # Calculate steps per epoch
        steps_per_epoch = max(1, len(images) // batch_size)
        
        print(f"\nTraining {model_name} for {epochs} epochs...")
        print(f"Steps per epoch: {steps_per_epoch}")
        
        history = model.model.fit(
            triplet_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
        )
    
    # Save the final model
    final_model_path = os.path.join(output_dir, f"{model_name}.h5")
    model.model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Plot training history and learning rate schedule
    plot_training_history(history, model_name, training_mode)
    
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

def plot_training_history(history, model_name, training_mode='supervised'):
    """Plot and save training history"""
    plt.figure(figsize=(12, 5))
    
    if training_mode == 'supervised':
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
    else:  # metric_learning
        # Only plot loss for metric learning
        plt.subplot(1, 1, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.title('Triplet Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'saved_figures/{model_name}_training_history.png')
    plt.show()

def create_triplet_data_from_supervised(data_collector, batch_size=32):
    """
    Create triplet training data from supervised data
    
    Args:
        data_collector: DataCollector instance
        batch_size: Batch size for triplet creation
        
    Returns:
        Tuple of (images_array, labels_array) for triplet generation
    """
    print("Converting supervised data to metric learning format...")
    
    # Load a small batch to get all unique samples
    train_ds, _, _ = data_collector.load_data(batch_size=1000, use_arcface=False)
    
    all_images = []
    all_labels = []
    
    # Extract all images and labels
    for batch in train_ds:
        if isinstance(batch, tuple):
            images, labels = batch
            all_images.extend(images.numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_images), np.array(all_labels)

def main():
    parser = argparse.ArgumentParser(description='Train ResNet models with multiple paradigms')
    parser.add_argument('--model_id', type=str, default='resnet_custom',
                        help='Model ID for saving the model')
    parser.add_argument('--training_mode', type=str, default='supervised', 
                        choices=['supervised', 'metric_learning'],
                        help='Training mode: supervised or metric_learning')
    parser.add_argument('--use_arcface', action='store_true',
                        help='Use ArcFace head (only for supervised mode)')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Base learning rate')
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
    
    print(f"Training ResNet with mode: {args.training_mode}")
    if args.training_mode == 'supervised' and args.use_arcface:
        print("Using ArcFace head for face recognition")
    
    # Initialize DataCollector
    data_collector = Datacollector(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path
    )
    
    # Get class names
    class_names = data_collector.get_class_names()
    if not class_names:
        print("Error: Unable to get class names from the data. Please check your data directories.")
        return
    
    print(f"Found {len(class_names)} classes: {class_names}")
    
    if args.training_mode == 'supervised':
        # Load the classification data
        train_ds, val_ds, test_ds = data_collector.load_data(
            batch_size=args.batch_size,
            use_arcface=args.use_arcface
        )
        
        # Train the model
        model, history = train_model(
            model_id=args.model_id,
            training_mode='supervised',
            use_arcface=args.use_arcface,
            train_data=train_ds,
            val_data=val_ds,
            test_data=test_ds,
            class_names=class_names,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
        
    else:  # metric_learning
        # Create triplet data from supervised data
        images, labels = create_triplet_data_from_supervised(data_collector, args.batch_size)
        
        print(f"Created triplet dataset with {len(images)} images")
        
        # Train the model
        model, history = train_model(
            model_id=args.model_id,
            training_mode='metric_learning',
            use_arcface=False,  # Not used in metric learning
            class_names=class_names,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            images=images,
            labels=labels
        )
    
    print(f"\nTraining completed! Check '{args.output_dir}' for saved models and 'saved_figures' for plots.")

if __name__ == "__main__":
    main()