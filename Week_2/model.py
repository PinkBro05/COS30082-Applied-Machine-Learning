# Metadata: Week 2 Assignment
# Description: This file contains the code for the model building and training for the Week 2 Assignment
# Result of model: Single Input Linear regression Average loss: 23.176805406701263
                # Multiple input Linear regression Average loss: 23.176153189869133
                # Single input DNN model Average loss: 15.306383263763538
                # Multiple input DNN model Average loss: 4.739821891922382
# Conclusion: For simple model such as Linear regression, there are not too many parameter to capture the information from data, because of that increase feature may only cause overfitting. On the other hand, DNN have a significant number of parameters, increase data can increase performance.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

def single_linear_regression_model(train_features, train_labels, test_features, test_labels, normalizer):
    # Single Linear Regression Model
    
    single_linear_model = tf.keras.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])

    single_linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')
    
    history = single_linear_model.fit(
        train_features['bmi'],
        train_labels,
        epochs=100,
        # Suppress logging.
        verbose=0,
        # Calculate validation results on 20% of the training data.
        validation_split=0.2)
    
    # hist = pd.DataFrame(history.history)
    # hist['epoch'] = history.epoch
    # print(hist.tail())
    
    # plot_loss(history)
    
    test_results = {}

    test_results['single_linear_model'] = single_linear_model.evaluate(
        test_features['bmi'],
        test_labels, verbose=0)
    
    average_loss = test_results['single_linear_model']/len(test_labels)
    print(f"Single Input Linear regression Average loss: {average_loss}")
    
    return single_linear_model

def multiple_linear_regression_model(train_features, train_labels, test_features, test_labels, normalizer):
    # Multiple Linear Regression Model
    
    multiple_linear_model = tf.keras.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])
    
    multiple_linear_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')
    
    history = multiple_linear_model.fit(
        train_features, train_labels, 
        epochs=100,
        # suppress logging
        verbose=0,
        # Calculate validation results on 20% of the training data
        validation_split=0.2)
    
    # hist = pd.DataFrame(history.history)
    # hist['epoch'] = history.epoch
    # print(hist.tail())
    
    # plot_loss(history)
    
    test_results = {}
    test_results['linear_model'] = multiple_linear_model.evaluate(
        test_features, test_labels, verbose=0)

    average_loss = test_results['linear_model']/len(test_labels)
    print(f"Multiple input Linear regression Average loss: {average_loss}")
    
    return multiple_linear_model
    
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(0.001))
    return model
    
def single_DNN_model(train_features, train_labels, test_features, test_labels, normalizer):
    
    single_dnn_model = build_and_compile_model(normalizer)
    
    history = single_dnn_model.fit(
    train_features['bmi'],
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)
    
    # hist = pd.DataFrame(history.history)
    # hist['epoch'] = history.epoch
    # print(hist.tail())
    
    # plot_loss(history)
    
    test_results = {}
    
    test_results['dnn_model'] = single_dnn_model.evaluate(
    test_features['bmi'], test_labels,
    verbose=0)
    
    average_loss = test_results['dnn_model']/len(test_labels)
    print(f"Single input DNN model Average loss: {average_loss}")
    
    return single_dnn_model
    
def multiple_DNN_model(train_features, train_labels, test_features, test_labels, normalizer):
    
    multiple_dnn_model = build_and_compile_model(normalizer)
    
    history = multiple_dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)
    
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
    
    plot_loss(history)
    
    test_results = {}
    
    test_results['dnn_model'] = multiple_dnn_model.evaluate(
    test_features, test_labels,
    verbose=0)
    
    average_loss = test_results['dnn_model']/len(test_labels)
    print(f"Multiple input DNN model Average loss: {average_loss}")
    return multiple_dnn_model

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [charges]')
    plt.legend()
    plt.grid(True)
    plt.show()

def prediction_visualization(model, test_features, test_labels):
    test_predictions = model.predict(test_features).flatten()
    
    # Scatter plot of true values vs. predictions
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [charges]')
    plt.ylabel('Predictions [charges]')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.title('True Values vs Predictions')
    
    # Histogram of prediction errors
    plt.subplot(1, 2, 2)
    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [charges]')
    plt.ylabel('Count')
    plt.title('Prediction Error Distribution')
    
    plt.tight_layout()
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
    
    # Data Normalization
    
    bmi = np.array(train_features['bmi'])
    bmi_normalizer = layers.Normalization(input_shape=[1,], axis=None)
    bmi_normalizer.adapt(bmi)
    
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))
    
    # Model Building
    # single_linear_regression_model(train_features, train_labels, test_features, test_labels, bmi_normalizer)
    # multiple_linear_regression_model(train_features, train_labels, test_features, test_labels, normalizer)
    # single_DNN_model(train_features, train_labels, test_features, test_labels, bmi_normalizer)
    best_model = multiple_DNN_model(train_features, train_labels, test_features, test_labels, normalizer)
    
    # Visualization prediction of best model
    prediction_visualization(best_model, test_features, test_labels)

if __name__ == "__main__":
    main()
