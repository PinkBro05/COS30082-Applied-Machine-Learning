import tensorflow as tf
from tensorflow.keras import  layers, models
import matplotlib.pyplot as plt

from ultis.data_collector import Datacollector

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
        else:
            self.shortcut = lambda x: x
            
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
        return self.activation(x)

class ResNet:
    def __init__(self, input_shape, num_classes):
        self.model = models.Sequential()
        self.model.add(layers.Input(shape=input_shape))
        
        # Initial Conv Layer
        self.model.add(layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu'))
        self.model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        
        # Residual Blocks
        self.model.add(ResidualBlock(64))
        self.model.add(ResidualBlock(64))
        self.model.add(ResidualBlock(64))
        
        self.model.add(ResidualBlock(128, stride=2))
        self.model.add(ResidualBlock(128))
        self.model.add(ResidualBlock(128))
        self.model.add(ResidualBlock(128))
        
        self.model.add(ResidualBlock(256, stride=2))
        self.model.add(ResidualBlock(256))
        self.model.add(ResidualBlock(256))
        self.model.add(ResidualBlock(256))
        self.model.add(ResidualBlock(256))
        self.model.add(ResidualBlock(256))
        
        self.model.add(ResidualBlock(512, stride=2))
        self.model.add(ResidualBlock(512))
        self.model.add(ResidualBlock(512))
                            
        # Final Layers
        self.model.add(layers.GlobalAveragePooling2D())
        self.model.add(layers.Dense(num_classes, activation='softmax'))
        
    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)
        
    def fit(self, train_ds, val_ds, epochs=60):
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
        image = tf.image.resize(image, (224, 224))
        image = tf.expand_dims(image, axis=0)
        image = image / 255.0
        
        # Make prediction
        predictions = self.model.predict(image)
        return predictions
    
    def save_model(self, model_path):
        self.model.save(model_path)
    
def main():
    # Load the data
    data_path = "Data"
    
    data = Datacollector(data_path)
    
    # Split the data into training and validation sets
    train_ds, val_ds = data.split_data(batch_size=32)
    
    input_shape = (224, 224, 3)
    num_classes = len(data.get_class_names())
    
    # Initialize the ResNet model
    resnet = ResNet(input_shape, num_classes)
    
    # print("Model Summary:")
    # resnet.model.summary()
    
    resnet.compile()
    
    # Train the model
    history = resnet.fit(train_ds, val_ds, epochs=1)
    
    resnet.plot_history(history)
    
    # Evaluate the model
    resnet.evaluate(val_ds)
    
    # Save the model
    resnet.save_model("models/resnet_model_1.h5")
    
if __name__ == "__main__":
    main()