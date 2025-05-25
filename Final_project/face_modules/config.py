"""
Configuration module for face recognition system.
Contains all configurable parameters for the face recognition system.
"""

# Threshold values
FACE_RECOGNITION_THRESHOLD = 1.2474 # Threshold for face recognition (adjusted for normalized embeddings)
EYE_ASPECT_RATIO_THRESHOLD = 0.25  # Threshold for detecting open/closed eyes

# Model paths
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, 'model_factory', 'new_models', 'embedding_euclidean.keras')
LANDMARK_PREDICTOR_PATH = os.path.join(BASE_DIR, 'model_factory', 'shape_predictor_68_face_landmarks.dat')

# Database files
EMBEDDINGS_FILE = os.path.join(BASE_DIR, 'db', 'embeddings.npy')
NAMES_FILE = os.path.join(BASE_DIR, 'db', 'names.txt')

# Facial landmarks indices
LEFT_EYE_LANDMARKS = list(range(36, 42))
RIGHT_EYE_LANDMARKS = list(range(42, 48))
