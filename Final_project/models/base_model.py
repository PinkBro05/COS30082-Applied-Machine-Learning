import tensorflow as tf
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Base class for all models"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        """
        Initialize the base model
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of classes for classification
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    @abstractmethod
    def build(self):
        """Build the model architecture - must be implemented by subclasses"""
        pass
    
    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        """Compile the model"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def fit(self, train_data, validation_data=None, epochs=30, callbacks=None, class_weights=None):
        """
        Train the model
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset
            epochs: Number of epochs to train
            callbacks: List of callbacks to use during training
            class_weights: Dictionary mapping class indices to weights for balanced training
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        return self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
    
    def evaluate(self, test_data, verbose=1):
        """
        Evaluate the model on test data
        
        Args:
            test_data: Test dataset
            verbose: Verbosity mode
            
        Returns:
            Tuple of (loss, accuracy)
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        return self.model.evaluate(test_data, verbose=verbose)
    
    def predict(self, input_data):
        """
        Make predictions on input data
        
        Args:
            input_data: Input data to make predictions on
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        return self.model.predict(input_data)
    
    def save(self, filepath):
        """
        Save the model to a file
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath, custom_objects=None):
        """
        Load a model from a file
        
        Args:
            filepath: Path to the saved model
            custom_objects: Dictionary mapping names (strings) to custom classes or functions
            
        Returns:
            Loaded model instance
        """
        instance = cls()
        instance.model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)
        instance.input_shape = instance.model.input_shape[1:4]
        instance.num_classes = instance.model.output_shape[1]
        return instance
    
    def plot_training_history(self, history):
        """
        Plot training history
        
        Args:
            history: History object returned by fit()
        """
        # Plot accuracy
        plt.figure(figsize=(12, 5))
        
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
        
        plt.tight_layout()
        plt.show()
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        self.model.summary()