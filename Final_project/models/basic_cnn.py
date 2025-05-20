import tensorflow as tf
from tensorflow.keras import layers, models
from models.base_model import BaseModel

class BasicCNN(BaseModel):
    """Basic CNN model for image classification"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        super().__init__(input_shape, num_classes)
    
    def build(self):
        """Build the Basic CNN model architecture"""
        model = models.Sequential()
        
        # First convolutional block
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Second convolutional block
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Third convolutional block
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Fourth convolutional block
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Flatten and dense layers
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))  # Add dropout for regularization
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        self.model = model
        return self.model