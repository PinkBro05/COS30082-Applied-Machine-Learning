import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt

class Datacollector:
    def __init__(self, data_path):
        self.data_dir = pathlib.Path(data_path)
                
        self.train_ds = None
        self.val_ds = None
            
    def split_data(self, batch_size=32, img_height=224, img_width=224):
        # split the dataset
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=1011,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=1011,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        
        # Standardize the data
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        normalized_train_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
        normalized_test_ds = self.val_ds.map(lambda x, y: (normalization_layer(x), y))
        
        return normalized_train_ds, normalized_test_ds
    
    def get_class_names(self):
        return self.train_ds.class_names
    
    def plot_sample_images(self, num_images=9):
        class_names = self.get_class_names()
        plt.figure(figsize=(10, 10))
        for images, labels in self.train_ds.take(1):
            for i in range(num_images):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
        plt.show()
    
def main():
    data_path = "Assignment/Data"
    
    data = Datacollector(data_path)
    
    train_ds, val_ds = data.split_data()

    data.plot_sample_images()
    
if __name__ == "__main__":
    main()