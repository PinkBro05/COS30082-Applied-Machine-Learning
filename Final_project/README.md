# Face Recognition Check-In System with Custom ResNet Training

This application is a comprehensive face recognition-based check-in system that combines computer vision techniques for face detection, eye blink detection for liveness verification, and face recognition. The system includes both a GUI interface for real-time face recognition and model training capabilities using custom ResNet architectures.

## Features

### GUI Application
- Real-time face detection and recognition
- Eye blink detection as a liveness check (anti-spoofing measure)
- User registration system with face embeddings
- Interactive GUI interface with camera integration

### Model Training
- Custom ResNet architecture implementation
- Support for both supervised and metric learning training paradigms
- ArcFace loss integration for improved face recognition
- Triplet loss with sliding window strategy for metric learning
- Comprehensive training and evaluation scripts

## Requirements

- Python 3.8+
- OpenCV
- dlib
- TensorFlow/Keras 2.x
- PIL (Pillow)
- NumPy
- scikit-learn
- matplotlib
- seaborn
- tkinter (included in standard Python)

## Setup

1. Install all required packages:

```bash
pip install opencv-python dlib tensorflow pillow numpy scikit-learn matplotlib seaborn
```

2. Download the required models:
   - `shape_predictor_68_face_landmarks.dat` (for facial landmarks detection)
   - Place pre-trained models in the `model_factory/` directory
   
3. Prepare your data:
   - For training: Place data in `data/classification_data/` with train/val/test splits
   - For verification: Use `data/verification_data/` with verification pairs

## Usage

### GUI Application (Face Recognition Check-In)

Run the main GUI application:

```bash
python main.py
```

#### Check-in Mode
1. Start the camera using the "Start Camera" button
2. The system will detect your face and attempt to recognize you
3. To complete check-in, blink your eyes (the system verifies liveness by detecting both open and closed eyes)
4. Once verified, your check-in will be confirmed

#### Registration Mode
1. Select "Registration" mode
2. Enter your name in the input field
3. Start the camera if not already running
4. Position your face in front of the camera
5. Click "Register Face" button
6. Your face will be registered in the system

### Model Training

Training is performed using the Jupyter Notebook `train.ipynb`. Open this notebook to:

1. Train a ResNet model for face classification with supervised learning
2. Use ArcFace loss for improved face recognition
3. Implement metric learning with triplet loss using sliding window strategy

#### Training Options

The notebook provides options for:

- **Model Identifier**: Name for saving your trained model
- **Training Mode**: Choose between 'supervised' or 'metric_learning'
- **ArcFace**: Enable/disable ArcFace head for supervised training
- **Hyperparameters**: Configure epochs, learning rate, batch size
- **Data Paths**: Specify training, validation, and test data locations

Run all cells in the notebook to train and save your model to the `model_factory/` directory.

### Model Evaluation

Model evaluation is integrated into the `train.ipynb` notebook, where you can:

- **Test Set Evaluation**: Evaluate a trained model on the test set
- **Sample Inference**: Visualize predictions on random samples
- **Face Verification**: Analyze ROC curves and verification performance

The notebook automatically generates evaluation metrics and visualizations in the `saved_figures/` directory, including:
- Classification metrics (accuracy, precision, recall, F1-score)
- ROC curves and AUC scores for face verification
- Confusion matrices
- Training history plots

## Project Structure

```
Final_project/
├── main.py                     # GUI application entry point
├── train.ipynb                 # Jupyter notebook for model training and evaluation
├── README.md                   # This file
├── face_modules/               # Core modules for face processing
│   ├── face_detector.py        # Face detection using dlib
│   ├── eye_detector.py         # Eye analysis and blink detection
│   ├── face_recognition.py     # Face recognition and embedding extraction
│   ├── check_in_system.py      # Integration of all modules for check-in workflow
│   ├── config.py               # Configuration settings
│   └── utils/                  # Utility functions for face analysis
├── utils/                      # Data processing utilities
│   └── indexer_algos.py        # Algorithm utilities
├── data/                       # Dataset storage
│   ├── classification_data/    # Training data (train/val/test splits)
│   ├── verification_data/      # Face verification pairs
│   ├── verification_pairs_test.txt  # Test verification pairs
│   └── verification_pairs_val.txt   # Validation verification pairs
├── model_factory/              # Pre-trained and saved models
│   ├── shape_predictor_68_face_landmarks.dat  # Facial landmarks model
│   └── *.h5                    # Trained model files
├── saved_figures/              # Training plots and visualizations
├── db/                         # Face embeddings and user data storage
└── __pycache__/               # Python cache files
```

## Model Architecture

The system uses a custom ResNet architecture (`ResNetCustom`) that supports:

- **Supervised Learning**: Traditional classification with cross-entropy or ArcFace loss
- **Metric Learning**: Triplet loss with sliding window strategy for better face embeddings
- **Flexible Training**: Switch between training paradigms without changing the base architecture

### Key Components

1. **ResidualBlock**: Custom residual blocks with batch normalization
2. **ArcFaceHead**: Angular margin loss head for improved face recognition
3. **TripletLossLayer**: Custom layer implementing triplet loss with margin
4. **Sliding Window Triplet Selection**: Systematic anchor-positive pairing strategy

## Training Features

- **Learning Rate Scheduling**: Warmup + Cosine decay scheduler
- **Early Stopping**: Prevents overfitting with patience mechanism
- **Model Checkpointing**: Saves best models during training
- **Data Augmentation**: Built-in augmentation for better generalization
- **Multiple Loss Functions**: Support for cross-entropy, ArcFace, and triplet loss

## Evaluation Metrics

The system provides comprehensive evaluation including:

- **Classification Metrics**: Accuracy, precision, recall, F1-score
- **Face Verification**: ROC curves, AUC scores, optimal thresholds
- **Confusion Matrices**: Detailed class-wise performance analysis
- **Visualization**: Training history plots and sample predictions

## Advanced Usage

### Custom Data Preparation

1. **Classification Data**: Organize images in class-specific folders:
   ```
   data/classification_data/
   ├── train_data/
   │   ├── person1/
   │   ├── person2/
   │   └── ...
   ├── val_data/
   └── test_data/
   ```

2. **Verification Pairs**: Create text files with image pairs and labels:
   ```
   person1/img1.jpg person1/img2.jpg 1
   person1/img1.jpg person2/img1.jpg 0
   ```

### Hyperparameter Tuning

Key hyperparameters to experiment with:

- **Learning Rate**: Start with 0.001, adjust based on convergence
- **Batch Size**: Balance between memory and gradient stability (16-64)
- **Margin (Triplet Loss)**: Typically 0.2-1.0
- **Scale (ArcFace)**: Usually 30-64
- **Epochs**: Monitor validation metrics to avoid overfitting

### Model Export for GUI

After training, models are automatically saved in `model_factory/`. The GUI application will use these trained models for face recognition.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU training
2. **Import Errors**: Ensure all dependencies are installed correctly
3. **Model Loading**: Check that custom layers are properly imported
4. **Camera Issues**: Verify camera permissions and OpenCV installation

### Performance Tips

- Use GPU acceleration for training (CUDA-enabled TensorFlow)
- Ensure good lighting conditions for face detection
- Regularly clean the face embeddings database
- Monitor training metrics to detect overfitting early

## Notes

## Notes

### For GUI Usage
- For optimal face detection, ensure good lighting conditions
- Keep your face within frame during check-in and registration
- The system requires both open and closed eye states to confirm liveness
- Face embeddings are stored in the `db/` directory for persistence

### For Model Training
- Training progress is automatically saved with plots in `saved_figures/`
- Models are saved in `model_factory/` with automatic checkpointing
- Use ArcFace loss for better face recognition performance
- Metric learning works well for few-shot learning scenarios
- Monitor GPU memory usage when training with large batch sizes

### Data Requirements
- Minimum 2 images per class for triplet learning
- At least 100 images per person for robust face recognition
- Balanced dataset recommended for better performance
- Images should be preprocessed to 64x64 pixels (automatic in data loader)

## License

This project is for educational purposes as part of the COS30082 Applied Machine Learning course.
