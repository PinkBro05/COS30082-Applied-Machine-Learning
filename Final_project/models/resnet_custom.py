import tensorflow as tf
from tensorflow.keras import layers, models
from models.base_model import BaseModel

class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, stride=1, activation='relu', name=None, **kwargs):
        super(ResidualBlock, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation_type = activation
        
        self.conv1 = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.activation = layers.Activation(activation)
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        
        # Shortcut connection
        if stride != 1:
            self.shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')
            self.bn3 = layers.BatchNormalization()
        else:
            self.shortcut = lambda x: x
            self.bn3 = lambda x: x
            
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "activation": self.activation_type,
        })
        return config

    def call(self, inputs):
        shortcut = self.shortcut(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += shortcut
        x = self.bn3(x)
        return self.activation(x)

class ResNetCustom(BaseModel):
    """Custom implementation of ResNet architecture"""
    
    def __init__(self, input_shape=(64, 64, 3), num_classes=10):
        super().__init__(input_shape, num_classes)
        
    def build(self, is_inference=False):
        """Build the ResNet model architecture"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial Conv Layer
        x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        
        # Residual Blocks
        # Stage 1
        for _ in range(3):
            x = ResidualBlock(64)(x)
        
        # Stage 2
        x = ResidualBlock(128, stride=2)(x)
        for _ in range(3):
            x = ResidualBlock(128)(x)
        
        # Stage 3
        x = ResidualBlock(256, stride=2)(x)
        for _ in range(5):
            x = ResidualBlock(256)(x)
        
        # Stage 4
        x = ResidualBlock(512, stride=2)(x)
        for _ in range(2):
            x = ResidualBlock(512)(x)
        
        # Final Layers
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(self.num_classes)(x)
        
        # If not inference, apply softmax activation else return logits as face embedding
        if not is_inference:
            outputs = layers.Activation('softmax')(outputs)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs, name="ResNetCustom")
        return self.model
    
    @classmethod
    def load(cls, filepath):
        """
        Load a ResNetCustom model from a file, handling the custom ResidualBlock layer
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        # Create custom_objects dictionary with the ResidualBlock class
        custom_objects = {'ResidualBlock': ResidualBlock}
        
        # Use the parent class's load method with the custom_objects
        return super().load(filepath, custom_objects=custom_objects)