import argparse
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from ultis.data_collector import Datacollector
from models import ResNet50Model, ResNetCustom, BasicCNN

def compute_class_weights(train_data):
    """
    Compute class weights to balance training
    
    Args:
        train_data: Training dataset
        
    Returns:
        Dictionary of class weights
    """
    # Extract all labels from the training data
    all_labels = []
    for images, labels in train_data.unbatch().as_numpy_iterator():
        all_labels.append(labels)
    
    all_labels = np.array(all_labels)
    
    # Compute class weights
    unique_classes = np.unique(all_labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=all_labels
    )
    
    # Create class weights dictionary
    class_weight_dict = {i: weight for i, weight in zip(unique_classes, class_weights)}
    
    print("Class weights:", class_weight_dict)
    return class_weight_dict

def train_model(
    model_type,
    train_data,
    val_data,
    test_data,
    class_names,
    epochs=60,
    learning_rate=0.001,
    output_dir='saved_models',
    use_class_weights=False,
    unfreeze_layers=None
):
    """
    Train a model
    
    Args:
        model_type: Type of model to train ('resnet50', 'resnet_custom', or 'basic_cnn')
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
    input_shape = (224, 224, 3)
    
    print(f"Creating {model_type} model with {num_classes} classes...")
    
    if model_type == 'resnet50':
        model = ResNet50Model(input_shape=input_shape, num_classes=num_classes)
        model.build()
        
        # Unfreeze layers for fine-tuning if specified
        if unfreeze_layers:
            model.unfreeze_layers(num_layers=unfreeze_layers)
    
    elif model_type == 'resnet_custom':
        model = ResNetCustom(input_shape=input_shape, num_classes=num_classes)
        model.build()
    
    elif model_type == 'basic_cnn':
        model = BasicCNN(input_shape=input_shape, num_classes=num_classes)
        model.build()
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Display model summary
    model.summary()
    
    # Define callbacks
    model_name = f"{model_type}_model{'_balanced' if use_class_weights else ''}"
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
    
    # Compute class weights if needed
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(train_data)
    
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
        class_weights=class_weights
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
    parser.add_argument('--model_type', type=str, default='resnet50', 
                        choices=['resnet50', 'resnet_custom', 'basic_cnn'],
                        help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=60, 
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size')
    parser.add_argument('--train_path', type=str, default='Data/train', 
                        help='Path to training data directory')
    parser.add_argument('--test_path', type=str, default='Data/test', 
                        help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='saved_models', 
                        help='Directory to save model and checkpoints')
    parser.add_argument('--use_class_weights', action='store_true', 
                        help='Use class weights for balanced training')
    parser.add_argument('--unfreeze_layers', type=int, default=None, 
                        help='Number of layers to unfreeze for fine-tuning (only for ResNet50)')
    
    args = parser.parse_args()
    
    # Initialize DataCollector
    data_collector = Datacollector(train_path=args.train_path, test_path=args.test_path)
    
    # Load and preprocess data
    # Use ResNet50 preprocessing for ResNet50 and ResNetCustom models
    use_resnet_preprocessing = args.model_type in ['resnet50', 'resnet_custom']
    
    train_data, val_data, test_data = data_collector.split_data(
        batch_size=args.batch_size, 
        img_height=224, 
        img_width=224,
        use_resnet_preprocessing=use_resnet_preprocessing
    )
    
    # Get class names
    class_names = data_collector.get_class_names()
    print(f"Classes (in alphabetical order): {class_names}")
    
    # Train the model
    train_model(
        model_type=args.model_type,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        class_names=class_names,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        use_class_weights=args.use_class_weights,
        unfreeze_layers=args.unfreeze_layers
    )
    
if __name__ == "__main__":
    main()