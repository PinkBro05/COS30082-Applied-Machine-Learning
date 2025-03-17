import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

def single_linear_regression_model(train_features, train_labels, test_features, test_labels):
    # Single Linear Regression Model
    
    # Data Normalization
    bmi = np.array(train_features['bmi'])
    bmi_normalizer = layers.Normalization(input_shape=[1,], axis=None)
    bmi_normalizer.adapt(bmi)
    
    insurance_model = tf.keras.Sequential([
        bmi_normalizer,
        layers.Dense(units=1)
    ])

    insurance_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')
    
    history = insurance_model.fit(
        train_features['bmi'],
        train_labels,
        epochs=100,
        # Suppress logging.
        verbose=0,
        # Calculate validation results on 20% of the training data.
        validation_split=0.2)
    
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
    
    plot_loss(history)
    
    test_results = {}

    test_results['insurance_model'] = insurance_model.evaluate(
        test_features['bmi'],
        test_labels, verbose=0)
    
    x = tf.linspace(0.0, 16, 53)
    y = insurance_model.predict(x)
    print(y)    

def multiple_linear_regression_model(train_features, train_labels, test_features, test_labels):
    # Multiple Linear Regression Model
    
    # Data Normalization
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))
    
    linear_model = tf.keras.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])
    
    linear_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')
    
    history = linear_model.fit(
        train_features, train_labels, 
        epochs=100,
        # suppress logging
        verbose=0,
        # Calculate validation results on 20% of the training data
        validation_split=0.2)
    
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
    
    plot_loss(history)
    
    test_results = {}
    test_results['linear_model'] = linear_model.evaluate(
        test_features, test_labels, verbose=0)
    
    print(test_results)

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [charges]')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Data Loading
    column_names = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
    raw_dataset = pd.read_csv("Week_2/Data/Medical_insurance.csv", names=column_names, sep=",", skiprows=1)
    dataset = raw_dataset.copy()
    
    # Data Preprocessing
    dataset = dataset.dropna() # Drop missing values
    dataset['smoker'] = dataset['smoker'].map({'no': 0, 'yes': 1}) # Encode smoker column
    dataset['sex'] = dataset['sex'].map({"female": 0, "male": 1}) # Encode sex column
    dataset = pd.get_dummies(dataset, columns=['region'], prefix='region') # One-hot encode region column    
    
    train_dataset = dataset.sample(frac=0.8, random_state=1011)
    test_dataset = dataset.drop(train_dataset.index)
    
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('charges')
    test_labels = test_features.pop('charges')
    
    # Model Building
    # single_linear_regression_model(train_features, train_labels, test_features, test_labels)
    multiple_linear_regression_model(train_features, train_labels, test_features, test_labels)

if __name__ == "__main__":
    main()
