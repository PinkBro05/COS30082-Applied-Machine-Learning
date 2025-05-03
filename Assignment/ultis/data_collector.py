import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input

class Datacollector:
    def __init__(self, train_path="Data/train", test_path="Data/test"):
        self.train_data_dir = pathlib.Path(train_path)
        self.test_data_dir = pathlib.Path(test_path)
                
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
            
    def split_data(self, batch_size=32, img_height=224, img_width=224, use_resnet_preprocessing=False):
        # split the training dataset into train and validation
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.train_data_dir,
            validation_split=0.2,
            subset="training",
            seed=1011,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.train_data_dir,
            validation_split=0.2,
            subset="validation",
            seed=1011,
            image_size=(img_height, img_width),
            batch_size=batch_size)
            
        # Load the test dataset (no split needed)
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
            # preprocess_input performs: x = x - mean and x = x / std
            # where mean and std are specific to the ResNet50 model
            print("Using ResNet50 preprocessing")
            
            def process_data(images, labels):
                return preprocess_input(images), labels
                
            processed_train_ds = self.train_ds.map(
                lambda x, y: (data_augmentation(x, training=True), y)
            ).map(process_data)
            
            processed_val_ds = self.val_ds.map(process_data)
            processed_test_ds = self.test_ds.map(process_data)
            
        else:
            # Standard normalization (0-1)
            print("Using standard normalization (0-1)")
            normalization_layer = tf.keras.layers.Rescaling(1./255)
            
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
        return self.train_ds.class_names
    
    def plot_sample_images(self, num_images=9, dataset_type="train"):
        """Plot sample images from the requested dataset (train, val, or test)"""
        class_names = self.get_class_names()
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
    data_path_train = "Data/train"
    data_path_test = "Data/test"
    
    # Check if the directories exist
    if not pathlib.Path(data_path_train).exists() or not pathlib.Path(data_path_test).exists():
        print("Train or test directories not found.")
        print("Please ensure Data/train and Data/test directories exist.")
        print("You can create them by running data_restructure.py first.")
        return
    
    data = Datacollector(train_path=data_path_train, test_path=data_path_test)
    
    train_ds, val_ds, test_ds = data.split_data()

    print("Class names:", data.get_class_names())
    
    print("\nShowing training samples:")
    data.plot_sample_images(dataset_type="train")
    
    print("\nShowing test samples:")
    data.plot_sample_images(dataset_type="test")
    
if __name__ == "__main__":
    main()