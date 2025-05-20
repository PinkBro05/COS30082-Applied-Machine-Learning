"""
Configuration module for face recognition system.
Contains all configurable parameters for the face recognition system.
"""

# Threshold values
FACE_RECOGNITION_THRESHOLD = 0.6  # Threshold for face recognition
EYE_ASPECT_RATIO_THRESHOLD = 0.25  # Threshold for detecting open/closed eyes

# Model paths
FACE_EMBEDDING_MODEL_PATH = 'model_factory/embedding_euclidean.keras'
LANDMARK_PREDICTOR_PATH = 'model_factory/shape_predictor_68_face_landmarks.dat'

# Database files
EMBEDDINGS_FILE = 'db/embeddings.npy'
NAMES_FILE = 'db/names.txt'

# Facial landmarks indices
LEFT_EYE_LANDMARKS = list(range(36, 42))
RIGHT_EYE_LANDMARKS = list(range(42, 48))
