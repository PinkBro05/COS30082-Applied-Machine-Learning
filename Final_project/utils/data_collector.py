import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input

class Datacollector:
    def __init__(self, train_path="data/classification_data/train_data", 
                 val_path="data/classification_data/val_data", 
                 test_path="data/classification_data/test_data"):
        """Initialize data collector with paths to classification data folders"""
        self.train_data_dir = pathlib.Path(train_path)
        self.val_data_dir = pathlib.Path(val_path)
        self.test_data_dir = pathlib.Path(test_path)
                
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
            
    def load_data(self, batch_size=32, img_height=64, img_width=64, use_resnet_preprocessing=True, use_arcface=False):
        """Load data from classification_data folder structure"""
        print("Loading classification data...")
        
        # Load the training dataset
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.train_data_dir,
            seed=1011,
            image_size=(img_height, img_width),
            batch_size=batch_size)
            
        # Load validation dataset
        print(f"Loading validation data from: {self.val_data_dir}")
        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.val_data_dir,
            seed=1011,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        
        # Load the test dataset
        self.test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.test_data_dir,
            image_size=(img_height, img_width),
            batch_size=batch_size)
            
        # Data augmentation for training data
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
        ])
        
        # Choose preprocessing method
        if use_resnet_preprocessing:
            # ResNet50 specific preprocessing
            print("Using ResNet50 preprocessing")
            
            def process_data(images, labels):
                return preprocess_input(images), labels
            
            if use_arcface:
                # For ArcFace - need to return images and labels as separate inputs
                def process_arcface_data(images, labels):
                    processed_images = preprocess_input(images)
                    return {'input_1': processed_images, 'label_input': labels}, labels
                
                processed_train_ds = self.train_ds.map(
                    lambda x, y: (data_augmentation(x, training=True), y)
                ).map(process_arcface_data)
                
                processed_val_ds = self.val_ds.map(process_arcface_data)
                processed_test_ds = self.test_ds.map(process_arcface_data)
                
            else:
                # Standard processing
                processed_train_ds = self.train_ds.map(
                    lambda x, y: (data_augmentation(x, training=True), y)
                ).map(process_data)
                
                processed_val_ds = self.val_ds.map(process_data)
                processed_test_ds = self.test_ds.map(process_data)
            
        else:
            # Standard normalization (0-1)
            print("Using standard normalization (0-1)")
            normalization_layer = tf.keras.layers.Rescaling(1./255)
            
            if use_arcface:
                # For ArcFace - need to return images and labels as separate inputs
                def process_arcface_norm_data(images, labels):
                    normalized_images = normalization_layer(images)
                    return {'input_1': normalized_images, 'label_input': labels}, labels
                
                processed_train_ds = self.train_ds.map(
                    lambda x, y: (data_augmentation(x, training=True), y)
                ).map(process_arcface_norm_data)
                
                processed_val_ds = self.val_ds.map(process_arcface_norm_data)
                processed_test_ds = self.test_ds.map(process_arcface_norm_data)
            
            else:
                # Standard processing 
                processed_train_ds = self.train_ds.map(
                    lambda x, y: (data_augmentation(x, training=True), y)
                ).map(lambda x, y: (normalization_layer(x), y))
                
                processed_val_ds = self.val_ds.map(
                    lambda x, y: (normalization_layer(x), y)
                )
                
                processed_test_ds = self.test_ds.map(
                    lambda x, y: (normalization_layer(x), y)
                )
        
        # Configure datasets for performance
        AUTOTUNE = tf.data.AUTOTUNE
        processed_train_ds = processed_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        processed_val_ds = processed_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        processed_test_ds = processed_test_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        return processed_train_ds, processed_val_ds, processed_test_ds
    
    def get_class_names(self):
        """Get class names from the training dataset"""
        return self.train_ds.class_names if self.train_ds else None
    
    def plot_sample_images(self, num_images=9, dataset_type="train"):
        """Plot sample images from the requested dataset (train, val, or test)"""
        class_names = self.get_class_names()
        if not class_names:
            print("No dataset loaded. Call load_data() first.")
            return
            
        plt.figure(figsize=(10, 10))
        
        if dataset_type == "train" and self.train_ds:
            dataset = self.train_ds
            title = "Training Dataset Samples"
        elif dataset_type == "val" and self.val_ds:
            dataset = self.val_ds
            title = "Validation Dataset Samples"
        elif dataset_type == "test" and self.test_ds:
            dataset = self.test_ds
            title = "Test Dataset Samples"
        else:
            print(f"Invalid dataset type: {dataset_type}")
            return
            
        plt.suptitle(title, fontsize=16)
        
        for images, labels in dataset.take(1):
            for i in range(min(num_images, len(images))):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
        plt.show()
    
def main():
    # Default paths for classification data
    classification_train_path = "data/classification_data/train_data"
    classification_val_path = "data/classification_data/val_data"
    classification_test_path = "data/classification_data/test_data"
    
    # Check if classification data directories exist
    if (not pathlib.Path(classification_train_path).exists() or
        not pathlib.Path(classification_val_path).exists() or
        not pathlib.Path(classification_test_path).exists()):
        print("Classification data directories not found.")
        print(f"Please ensure {classification_train_path}, {classification_val_path}, and {classification_test_path} directories exist.")
        return
        
    # Create a DataCollector instance with the classification data paths
    data = Datacollector(
        train_path=classification_train_path,
        val_path=classification_val_path,
        test_path=classification_test_path
    )
    
    # Load the classification data
    train_ds, val_ds, test_ds = data.load_data()
    
    # Display info about the loaded data
    print("Classification dataset class names:", data.get_class_names())
    print(f"Number of classes: {len(data.get_class_names())}")
    
    # Visualize samples from each dataset
    print("\nShowing training samples:")
    data.plot_sample_images(dataset_type="train")
    
    print("\nShowing validation samples:")
    data.plot_sample_images(dataset_type="val")
    
    print("\nShowing test samples:")
    data.plot_sample_images(dataset_type="test")
    
if __name__ == "__main__":
    main()