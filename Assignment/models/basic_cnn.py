import tensorflow as tf
from tensorflow.keras import  layers, models
import matplotlib.pyplot as plt

from ultis.data_collector import Datacollector

class BasicCNN:
    def __init__(self, input_shape, num_classes):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(num_classes))
        
    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)
    
    def fit(self, train_ds, val_ds, epochs=10):
        history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        return history
    
    def evaluate(self, test_ds):
        test_loss, test_acc = self.model.evaluate(test_ds)
        print(f'Test accuracy: {test_acc}')
        return test_loss, test_acc
    
    def plot_history(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'b', label='Training accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()
        
    def predict(self, image):
        # Preprocess the image
        image = tf.image.resize(image, (180, 180))
        image = tf.expand_dims(image, axis=0)
        image = image / 255.0
        
        # Make prediction
        predictions = self.model.predict(image)
        predicted_class = tf.argmax(predictions[0]).numpy()
        return predicted_class

def main():
    
    # Initialize the data collector with the path to your dataset
    data_path = "./Data"
    data = Datacollector(data_path)
    
    # Split the dataset into training and validation sets
    train_ds, test_ds = data.split_data(batch_size=32, img_height=180, img_width=180)
    
    # Initialize the BasicCNN model
    input_shape = (180, 180, 3)  # Change this according to your image size
    num_classes = len(data.get_class_names())
    model = BasicCNN(input_shape, num_classes)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(train_ds, test_ds, epochs=10)
    
    # Evaluate the model
    model.evaluate(test_ds)
    
    # Plot training history
    model.plot_history(history)
    

if __name__ == "__main__":
    main()