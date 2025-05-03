import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from ultis.data_collector import Datacollector

def main():
    # Set the path to the dataset directory
    train_path = 'Data/train'  # Replace with your dataset path
    test_path = 'Data/test'  # Replace with your test dataset path
    
    # Initialize DataCollector
    data_collector = Datacollector(train_path=train_path, test_path=test_path)

    # Load and preprocess data with ResNet50-specific preprocessing
    train_data, val_data, test_data = data_collector.split_data(
        batch_size=32, 
        img_height=224, 
        img_width=224,
        use_resnet_preprocessing=True  # Enable ResNet50 preprocessing
    )

    # Get the number of classes
    class_names = data_collector.get_class_names()
    num_classes = len(class_names)
    
    # Define the base model (ResNet50)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top of the base model
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Define model checkpoint directory
    checkpoint_dir = 'models/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Define callbacks
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
            filepath=os.path.join(checkpoint_dir, 'best_resnet_model.h5'),
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
        optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.001,
            rho=0.95,  # No decay for Adadelta
            epsilon=1e-07,  # Epsilon for numerical stability
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model with callbacks
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=60,
        callbacks=callbacks,
        verbose=1
    )

    # Save the final model (best weights already saved by ModelCheckpoint)
    model.save('resnet_fine_tuned_final.h5')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    main()