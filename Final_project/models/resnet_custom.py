import tensorflow as tf
from tensorflow.keras import layers, models
from models.base_model import BaseModel

class ArcFaceHead(layers.Layer):
    def __init__(self, num_classes, scale=30.0, margin=0.5, name=None, **kwargs):
        super(ArcFaceHead, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.w = None

    def build(self, input_shape):
        # Check input shape format
        if len(input_shape) != 2:
            raise ValueError(f"ArcFaceHead expects 2 inputs: [features, labels], got {len(input_shape)}")
        
        # Get the feature dimension from input
        feature_dim = input_shape[0][-1]
        
        # Initialize weights matrix for class prototypes
        self.w = self.add_weight(
            name="prototype_embeddings",
            shape=(self.num_classes, feature_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        super(ArcFaceHead, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "scale": self.scale,
            "margin": self.margin,
        })
        return config

    def call(self, inputs):
        # Check input format and unpack inputs: features and labels
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError(f"ArcFaceHead expects inputs as [features, labels], got {inputs}")
        
        features, labels = inputs
        
        # Ensure labels are properly shaped
        labels = tf.reshape(labels, [-1])  # Flatten to 1D
        
        # L2 normalize features and weights
        features_norm = tf.nn.l2_normalize(features, axis=1)
        weights_norm = tf.nn.l2_normalize(self.w, axis=1)
        
        # Compute cosine similarity
        cosine = tf.matmul(features_norm, weights_norm, transpose_b=True)
        
        # Clip cosine values to avoid numerical instability
        cosine = tf.clip_by_value(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Get target logits (from one-hot)
        one_hot_labels = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes)
        
        # Calculate arcface logits
        sine = tf.sqrt(1.0 - tf.square(cosine))
        cos_m = tf.cos(self.margin)
        sin_m = tf.sin(self.margin)
        phi = cosine * cos_m - sine * sin_m
        
        # Apply margin only to positive logits (corresponding to the true class)
        updated_target_logits = tf.where(one_hot_labels > 0, phi, cosine)
        
        # Scale logits
        logits = updated_target_logits * self.scale
        
        return logits

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
        self.is_inference_model = False
        
    def build(self, is_inference=False, is_acrf=False):
        """Build the ResNet model architecture"""
        self.is_inference_model = is_inference
        inputs = layers.Input(shape=self.input_shape, name='input_1')
        
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
        features = layers.Dense(512)(x)  # Feature embedding layer (512 dimensions)
        
        if is_inference:
            # For inference, only use the image input and return L2 normalized features
            # L2 normalization is important for consistent similarity calculations in face recognition
            normalized_features = tf.nn.l2_normalize(features, axis=1, name='l2_normalization')
            self.model = models.Model(inputs=inputs, outputs=normalized_features, name="ResNetCustom_Inference")
        else:
            # For training, use both image and label inputs
            if is_acrf:
                # Add ArcFace head for face recognition
                # Create input for labels (to be used during training)
                label_input = layers.Input(shape=(1,), name='label_input', dtype=tf.int32)
                
                outputs = ArcFaceHead(self.num_classes)([features, label_input])
                
                # Create model with two inputs: image and labels
                self.model = models.Model(inputs=[inputs, label_input], outputs=outputs, name="ResNetCustom")
            else:
                # Normarl classification head
                outputs = layers.Dense(self.num_classes)(x)
                
                outputs = layers.Activation('softmax')(outputs)
                
                self.model = models.Model(inputs=inputs, outputs=outputs, name="ResNetCustom_Classification")
            
        
        return self.model
        
    @classmethod
    def load(cls, filepath):
        """
        Load a ResNetCustom model from a file, handling the custom ResidualBlock and ArcFaceHead layers
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        # Create custom_objects dictionary with the custom layer classes
        custom_objects = {
            'ResidualBlock': ResidualBlock,
            'ArcFaceHead': ArcFaceHead
        }
        
        # Use the parent class's load method with the custom_objects
        return super().load(filepath, custom_objects=custom_objects)
    
    def evaluate(self, test_data, verbose=1):
        """
        Evaluate the model on test data, handling both training and inference models
        
        Args:
            test_data: Test dataset
            verbose: Verbosity mode
            
        Returns:
            Tuple of (loss, accuracy)
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        if self.is_inference_model:
            # For inference model, we can't directly evaluate accuracy
            # This would need a custom evaluation approach
            raise ValueError("Cannot directly evaluate an inference model. Use the training model for evaluation.")
        else:
            # For training model with ArcFace, test_data should already be properly formatted
            return self.model.evaluate(test_data, verbose=verbose)
    
    def fit(self, train_data, validation_data=None, epochs=30, callbacks=None, class_weights=None):
        """
        Train the model, handling both training and inference models
        
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
        
        if self.is_inference_model:
            raise ValueError("Cannot train an inference model. Use a training model instead.")
        
        # For training model with ArcFace, train_data should already be properly formatted
        return self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
    
    def predict(self, input_data):
        """
        Make predictions on input data, handling both training and inference models
        
        Args:
            input_data: Input data to make predictions on. Can be images, 
                        a tuple of (images, labels), or a dataset
            
        Returns:
            Predictions: Face embeddings for inference model or class probabilities for training model
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        if self.is_inference_model:
            # For inference model, we only need the images
            if isinstance(input_data, tuple) or isinstance(input_data, list):
                # If it's a tuple/list of (images, labels), just use the images
                images = input_data[0]
                return self.model.predict(images)
            else:
                # It's just images or a dataset that yields images
                return self.model.predict(input_data)
        else:
            # For training model, we need both images and labels
            if hasattr(input_data, 'map') and callable(getattr(input_data, 'map')):
                # It's likely a dataset, let the model handle it
                return self.model.predict(input_data)
            elif isinstance(input_data, tuple) or isinstance(input_data, list):
                # If it's already a tuple/list of (images, labels)
                images, labels = input_data
                return self.model.predict({'input_1': images, 'label_input': labels})
            else:
                # Just images, generate dummy labels
                batch_size = tf.shape(input_data)[0]
                dummy_labels = tf.zeros((batch_size, 1), dtype=tf.int32)
                return self.model.predict({'input_1': input_data, 'label_input': dummy_labels})