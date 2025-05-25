import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from models.base_model import BaseModel

# Triplet Loss Layer for Metric Learning
class TripletLossLayer(layers.Layer):
    def __init__(self, margin=0.5, name=None, **kwargs):
        super(TripletLossLayer, self).__init__(name=name, **kwargs)
        self.margin = margin

    def get_config(self):
        config = super().get_config()
        config.update({"margin": self.margin})
        return config

    def call(self, inputs):
        # inputs should be [anchor, positive, negative] embeddings
        if not isinstance(inputs, list) or len(inputs) != 3:
            raise ValueError(f"TripletLossLayer expects inputs as [anchor, positive, negative], got {inputs}")
        
        anchor, positive, negative = inputs
        
        # Compute distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        # Triplet loss
        basic_loss = pos_dist - neg_dist + self.margin
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        
        # Add loss to layer
        self.add_loss(loss)
        
        # Return anchor embeddings for model output
        return anchor

# Face recognition specific head for ArcFace loss
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

# Residual Block for ResNet
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

# ResNet 18 with Multiple Training Paradigms
class ResNetCustom(BaseModel):
    """Custom implementation of ResNet architecture supporting supervised and metric learning"""
    
    def __init__(self, input_shape=(64, 64, 3), num_classes=10):
        super().__init__(input_shape, num_classes)
        self.is_inference_model = False
        self.training_mode = 'supervised'  # 'supervised' or 'metric_learning'
        
    def build(self, is_inference=False, training_mode='supervised', use_arcface=False):
        """
        Build the ResNet model architecture
        
        Args:
            is_inference: If True, build model for inference (returns normalized features)
            training_mode: 'supervised' for classification or 'metric_learning' for triplet/siamese
            use_arcface: If True and training_mode='supervised', use ArcFace head
        """
        self.is_inference_model = is_inference
        self.training_mode = training_mode
        
        if training_mode == 'metric_learning':
            return self._build_siamese_model(is_inference)
        else:
            return self._build_supervised_model(is_inference, use_arcface)
    
    def _build_backbone(self, input_tensor):
        """Build the ResNet backbone (shared for all configurations)"""
        # Initial Conv Layer
        x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_tensor)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        
        # Residual Blocks
        # Stage 1
        for _ in range(2):
            x = ResidualBlock(64)(x)
        
        # Stage 2
        x = ResidualBlock(128, stride=2)(x)
        for _ in range(2):
            x = ResidualBlock(128)(x)
        
        # Stage 3
        x = ResidualBlock(256, stride=2)(x)
        for _ in range(2):
            x = ResidualBlock(256)(x)
        
        # Stage 4
        x = ResidualBlock(512, stride=2)(x)
        for _ in range(2):
            x = ResidualBlock(512)(x)
        
        # Final Layers
        x = layers.GlobalAveragePooling2D()(x)
        features = layers.Dense(512, name='feature_embeddings')(x)  # Feature embedding layer (512 dimensions)
        
        return features
    
    def _build_supervised_model(self, is_inference=False, use_arcface=False):
        """Build model for supervised learning (classification)"""
        inputs = layers.Input(shape=self.input_shape, name='input_1')
        features = self._build_backbone(inputs)
        
        if is_inference:
            # For inference, return L2 normalized features
            normalized_features = tf.nn.l2_normalize(features, axis=1, name='l2_normalization')
            self.model = models.Model(inputs=inputs, outputs=normalized_features, name="ResNetCustom_Inference")
        else:
            if use_arcface:
                # Add ArcFace head for face recognition
                label_input = layers.Input(shape=(1,), name='label_input', dtype=tf.int32)
                outputs = ArcFaceHead(self.num_classes)([features, label_input])
                self.model = models.Model(inputs=[inputs, label_input], outputs=outputs, name="ResNetCustom_ArcFace")
            else:
                # Normal classification head
                outputs = layers.Dense(self.num_classes)(features)
                outputs = layers.Activation('softmax')(outputs)
                self.model = models.Model(inputs=inputs, outputs=outputs, name="ResNetCustom_Classification")
        
        return self.model
    
    def _build_siamese_model(self, is_inference=False):
        """Build model for metric learning (Siamese/Triplet)"""
        if is_inference:
            # For inference, just return the embedding model
            inputs = layers.Input(shape=self.input_shape, name='input_1')
            features = self._build_backbone(inputs)
            normalized_features = tf.nn.l2_normalize(features, axis=1, name='l2_normalization')
            self.model = models.Model(inputs=inputs, outputs=normalized_features, name="ResNetCustom_Siamese_Inference")
        else:
            # For training, create triplet inputs
            anchor_input = layers.Input(shape=self.input_shape, name='anchor_input')
            positive_input = layers.Input(shape=self.input_shape, name='positive_input')
            negative_input = layers.Input(shape=self.input_shape, name='negative_input')
            
            # Shared backbone
            anchor_features = self._build_backbone(anchor_input)
            positive_features = self._build_backbone(positive_input)
            negative_features = self._build_backbone(negative_input)
            
            # L2 normalize embeddings
            anchor_norm = tf.nn.l2_normalize(anchor_features, axis=1)
            positive_norm = tf.nn.l2_normalize(positive_features, axis=1)
            negative_norm = tf.nn.l2_normalize(negative_features, axis=1)
            
            # Triplet loss layer
            output = TripletLossLayer(margin=0.5)([anchor_norm, positive_norm, negative_norm])
            
            self.model = models.Model(
                inputs=[anchor_input, positive_input, negative_input], 
                outputs=output, 
                name="ResNetCustom_Triplet"
            )
        
        return self.model
    
    @classmethod
    def load(cls, filepath):
        """
        Load a ResNetCustom model from a file, handling custom layers
        """
        custom_objects = {
            'ResidualBlock': ResidualBlock,
            'ArcFaceHead': ArcFaceHead,
            'TripletLossLayer': TripletLossLayer
        }
        return super().load(filepath, custom_objects=custom_objects)
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model based on training mode"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        if self.training_mode == 'supervised':
            if 'ArcFace' in self.model.name:
                # For ArcFace, use sparse categorical crossentropy
                self.model.compile(
                    optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
            else:
                # For normal classification
                self.model.compile(
                    optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
        else:  # metric_learning
            # For triplet loss, loss is computed in the layer
            self.model.compile(optimizer=optimizer)
    
    def compute_similarity(self, embedding1, embedding2, metric='cosine'):
        """
        Compute similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: 'cosine' or 'euclidean'
            
        Returns:
            Similarity score
        """
        if metric == 'cosine':
            # Cosine similarity
            dot_product = tf.reduce_sum(embedding1 * embedding2, axis=1)
            norm1 = tf.norm(embedding1, axis=1)
            norm2 = tf.norm(embedding2, axis=1)
            similarity = dot_product / (norm1 * norm2)
            return similarity
        
        elif metric == 'euclidean':
            # Euclidean distance (convert to similarity)
            distance = tf.norm(embedding1 - embedding2, axis=1)
            # Convert distance to similarity (closer = higher similarity)
            similarity = 1.0 / (1.0 + distance)
            return similarity
        else:
            raise ValueError("Metric must be 'cosine' or 'euclidean'")
    
    def evaluate_face_verification(self, verification_pairs, labels, metric='cosine', plot_roc=True):
        """
        Evaluate face verification performance using ROC curve and AUC
        
        Args:
            verification_pairs: List of tuples (image1, image2) for verification
            labels: List of ground truth labels (1 for same person, 0 for different)
            metric: Similarity metric to use ('cosine' or 'euclidean')
            plot_roc: Whether to plot ROC curve
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        if not self.is_inference_model:
            raise ValueError("Use inference model for face verification evaluation")
        
        similarities = []
        
        print("Computing similarities for verification pairs...")
        for i, (img1, img2) in enumerate(verification_pairs):
            if i % 100 == 0:
                print(f"Processing pair {i+1}/{len(verification_pairs)}")
            
            # Get embeddings
            emb1 = self.model.predict(np.expand_dims(img1, axis=0), verbose=0)
            emb2 = self.model.predict(np.expand_dims(img2, axis=0), verbose=0)
            
            # Compute similarity
            sim = self.compute_similarity(emb1, emb2, metric=metric)
            similarities.append(float(sim.numpy()))
        
        similarities = np.array(similarities)
        labels = np.array(labels)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's index)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
        
        # Compute metrics at optimal threshold
        predictions = (similarities >= optimal_threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        
        # True positive rate and false positive rate at optimal threshold
        tpr_optimal = tpr[optimal_idx]
        fpr_optimal = fpr[optimal_idx]
        
        # Compute precision and recall
        true_positives = np.sum((predictions == 1) & (labels == 1))
        false_positives = np.sum((predictions == 1) & (labels == 0))
        false_negatives = np.sum((predictions == 0) & (labels == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'auc': roc_auc,
            'optimal_threshold': optimal_threshold,
            'accuracy': accuracy,
            'tpr': tpr_optimal,
            'fpr': fpr_optimal,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'similarities': similarities,
            'fpr_curve': fpr,
            'tpr_curve': tpr,
            'thresholds': thresholds
        }
        
        print(f"\n=== Face Verification Results ({metric} similarity) ===")
        print(f"AUC: {roc_auc:.4f}")
        print(f"Optimal Threshold: {optimal_threshold:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        print(f"TPR: {tpr_optimal:.4f}")
        print(f"FPR: {fpr_optimal:.4f}")
        
        if plot_roc:
            self.plot_roc_curve(fpr, tpr, roc_auc, metric)
        
        return results
    
    def plot_roc_curve(self, fpr, tpr, auc_score, metric='cosine'):
        """Plot ROC curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Face Verification ({metric} similarity)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def create_triplet_dataset(self, images, labels, batch_size=32):
        """
        Create triplet dataset for metric learning using sliding window strategy
        
        Args:
            images: Array of images
            labels: Array of corresponding labels
            batch_size: Batch size for the dataset
            
        Returns:
            TensorFlow dataset with triplets
        """
        # Pre-organize data by class for efficient access
        class_indices = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            class_indices[label] = np.where(labels == label)[0]
        
        def triplet_generator():
            while True:
                triplet_batch = []
                
                # For each class, create triplets using sliding window
                for label in unique_labels:
                    indices = class_indices[label]
                    
                    # Skip if class has less than 2 samples
                    if len(indices) < 2:
                        continue
                    
                    # Apply sliding window with size=2, stride=2
                    for i in range(0, len(indices) - 1, 2):
                        if i + 1 < len(indices):
                            # Get anchor and positive from sliding window
                            anchor_idx = indices[i]
                            positive_idx = indices[i + 1]
                            
                            # Get all negative classes
                            negative_labels = [l for l in unique_labels if l != label]
                            if len(negative_labels) == 0:
                                continue
                            
                            # Randomly select a negative class and then a random sample from that class
                            negative_label = np.random.choice(negative_labels)
                            negative_candidates = class_indices[negative_label]
                            negative_idx = np.random.choice(negative_candidates)
                            
                            triplet_batch.append((anchor_idx, positive_idx, negative_idx))
                            
                            # Yield batch when we have enough triplets
                            if len(triplet_batch) >= batch_size:
                                for anchor_idx, positive_idx, negative_idx in triplet_batch[:batch_size]:
                                    yield (
                                        {
                                            'anchor_input': images[anchor_idx],
                                            'positive_input': images[positive_idx],
                                            'negative_input': images[negative_idx]
                                        },
                                        images[anchor_idx]  # Dummy target (loss computed in layer)
                                    )
                                triplet_batch = triplet_batch[batch_size:]
                
                # Yield remaining triplets if any
                if triplet_batch:
                    for anchor_idx, positive_idx, negative_idx in triplet_batch:
                        yield (
                            {
                                'anchor_input': images[anchor_idx],
                                'positive_input': images[positive_idx],
                                'negative_input': images[negative_idx]
                            },
                            images[anchor_idx]  # Dummy target (loss computed in layer)
                        )
        
        dataset = tf.data.Dataset.from_generator(
            triplet_generator,
            output_signature=(
                {
                    'anchor_input': tf.TensorSpec(shape=self.input_shape, dtype=tf.float32),
                    'positive_input': tf.TensorSpec(shape=self.input_shape, dtype=tf.float32),
                    'negative_input': tf.TensorSpec(shape=self.input_shape, dtype=tf.float32)
                },
                tf.TensorSpec(shape=self.input_shape, dtype=tf.float32)
            )
        )
        
        return dataset.batch(1)  # Already batched in generator
    
    def evaluate(self, test_data, verbose=1):
        """
        Evaluate the model on test data
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        if self.is_inference_model:
            raise ValueError("Cannot directly evaluate an inference model. Use the training model for evaluation.")
        
        if self.training_mode == 'metric_learning':
            # For metric learning, evaluation is typically done through face verification
            print("For metric learning models, use evaluate_face_verification() method instead.")
            return None
        else:
            return self.model.evaluate(test_data, verbose=verbose)
    
    def fit(self, train_data, validation_data=None, epochs=30, callbacks=None, class_weights=None):
        """
        Train the model
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        if self.is_inference_model:
            raise ValueError("Cannot train an inference model. Use a training model instead.")
        
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
        Make predictions on input data
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        if self.is_inference_model:
            # For inference model, just return embeddings
            return self.model.predict(input_data)
        else:
            if self.training_mode == 'supervised':
                if 'ArcFace' in self.model.name:
                    # Handle ArcFace model with two inputs
                    if isinstance(input_data, tuple) or isinstance(input_data, list):
                        images, labels = input_data
                        return self.model.predict({'input_1': images, 'label_input': labels})
                    else:
                        # Generate dummy labels for prediction
                        batch_size = tf.shape(input_data)[0]
                        dummy_labels = tf.zeros((batch_size, 1), dtype=tf.int32)
                        return self.model.predict({'input_1': input_data, 'label_input': dummy_labels})
                else:
                    # Normal classification model
                    return self.model.predict(input_data)
            else:
                # For triplet model in training mode
                return self.model.predict(input_data)