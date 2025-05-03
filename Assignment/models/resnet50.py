import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from models.base_model import BaseModel

class ResNet50Model(BaseModel):
    """ResNet50 model from Keras applications"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=10, weights='imagenet', include_top=False):
        super().__init__(input_shape, num_classes)
        self.weights = weights
        self.include_top = include_top
        self.base_model = None
    
    def build(self):
        """Build the ResNet50 model architecture"""
        # Load the ResNet50 base model
        self.base_model = ResNet50(
            weights=self.weights,
            include_top=self.include_top,
            input_shape=self.input_shape
        )
        
        # Freeze the base model layers
        for layer in self.base_model.layers:
            layer.trainable = False
        
        # Add custom layers on top
        x = self.base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create the model
        self.model = models.Model(inputs=self.base_model.input, outputs=outputs, name="ResNet50Finetuned")
        return self.model
    
    def unfreeze_layers(self, num_layers=10):
        """
        Unfreeze the last n layers of the base model for fine-tuning
        
        Args:
            num_layers: Number of layers to unfreeze, counting from the end
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        # Find ResNet layers to unfreeze
        for layer in self.base_model.layers[-num_layers:]:
            layer.trainable = True
            
        print(f"Unfrozen the last {num_layers} layers of the base model for fine-tuning")