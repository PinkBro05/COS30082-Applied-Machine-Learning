# Animal Classification with Deep Learning

This project implements and evaluates several deep learning models for multi-class animal classification using TensorFlow/Keras. The codebase follows object-oriented programming principles for maintainability and extensibility.

## Directory Structure

```
├── Data/                      # Dataset directory
│   ├── test/                  # Test dataset with class subfolders
│   │   ├── Amphibia/
│   │   ├── Animalia/
│   │   ├── Arachnida/
│   │   ├── Aves/
│   │   ├── Fungi/
│   │   ├── Insecta/
│   │   ├── Mammalia/
│   │   ├── Mollusca/
│   │   ├── Plantae/
│   │   └── Reptilia/
│   └── train/                 # Training dataset with class subfolders
│       ├── Amphibia/
│       ├── Animalia/
│       └── ...                # Same class structure as test
│
├── models/                    # Model definitions using OOP
│   ├── __init__.py           # Package exports
│   ├── base_model.py         # Abstract base class for all models
│   ├── basic_cnn.py          # Simple CNN implementation
│   ├── resnet_custom.py      # From-scratch Resnet50 model
│   └── resnet50.py           # Pre-trained ResNet50 model
│
├── saved_models/             # Saved model weights
│
├── ultis/                    # Utility functions
│   └── data_collector.py     # Data loading and preprocessing
│
├── evaluate.py               # Script for model evaluation
├── inference.py              # Script for making predictions
└── train.py                  # Script for model training
```

## Models

The project implements three different model architectures:

1. **BasicCNN**: A simple convolutional neural network with multiple convolutional layers.
2. **ResNetCustom**: A custom implementation of ResNet architecture with residual blocks.
3. **ResNet50Model**: Uses pre-trained ResNet50 from Keras applications with fine-tuning capabilities.

All models inherit from a common `BaseModel` abstract class that provides standard functionality for training, evaluation, and inference.

## Usage

### Training a Model

The `train.py` script provides a flexible command-line interface for training models:

```bash
# Train ResNet50 with default parameters
python train.py --model_type resnet50

# Train with class weights for balanced training
python train.py --model_type resnet50 --use_class_weights

# Fine-tune ResNet50 by unfreezing the last 20 layers
python train.py --model_type resnet50 --unfreeze_layers 20 --learning_rate 0.0001

# Train BasicCNN with custom parameters
python train.py --model_type basic_cnn --epochs 100 --batch_size 64 --learning_rate 0.001
```

Parameters:
- `--model_type`: Model architecture to use (`resnet50`, `resnet_custom`, or `basic_cnn`)
- `--epochs`: Number of training epochs
- `--learning_rate`: Learning rate for optimizer
- `--batch_size`: Batch size for training
- `--train_path`: Path to training data directory
- `--test_path`: Path to test data directory
- `--output_dir`: Directory to save model and checkpoints
- `--use_class_weights`: Use class weights for balanced training
- `--unfreeze_layers`: Number of layers to unfreeze for fine-tuning (only for ResNet50)

### Evaluating a Model

The `evaluate.py` script provides comprehensive model evaluation capabilities:

```bash
# Standard evaluation
python evaluate.py --model_path saved_models/best_resnet_model_balanced.h5 --model_type resnet50

# Analyze potential class bias in the model
python evaluate.py --model_path saved_models/best_resnet_model_balanced.h5 --model_type resnet50 --analyze_bias
```

Parameters:
- `--model_path`: Path to the saved model file
- `--model_type`: Type of model (`resnet50`, `resnet_custom`, `basic_cnn`, or `generic`)
- `--test_data_path`: Path to test data directory
- `--analyze_bias`: Analyze potential class bias in the model

### Making Predictions

The `inference.py` script lets you use a trained model to make predictions:

```bash
# Predict using a saved model on a random test image
python inference.py --model_path saved_models/best_resnet_model_balanced.h5 --model_type resnet50

# Predict a specific image
python inference.py --model_path saved_models/best_resnet_model_balanced.h5 --image_path path/to/your/image.jpg
```

Parameters:
- `--model_path`: Path to the saved model file
- `--model_type`: Type of model (`resnet50`, `resnet_custom`, or `basic_cnn`)
- `--image_path`: Path to specific image to predict (optional)
- `--data_path`: Path to test data directory (used if image_path is not provided)

## Class Bias Analysis

The project includes functionality to analyze potential bias in model predictions related to class order. This is particularly useful for identifying if certain classes consistently receive higher prediction probabilities regardless of the true class.

## Data Preprocessing

The `data_collector.py` module handles data loading, preprocessing, and augmentation:

- Splits training data into train and validation sets
- Applies data augmentation (random flips, rotations, zooms)
- Supports standard normalization (0-1) or ResNet-specific preprocessing
- Configures datasets for performance with caching and prefetching

## Requirements

- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
- pandas
- seaborn