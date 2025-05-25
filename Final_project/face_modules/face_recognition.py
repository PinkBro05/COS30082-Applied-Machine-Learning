"""
Face recognition module.
Provides functionality for recognizing faces using embeddings.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from face_modules.utils.math_utils import distance
from face_modules.config import (
    FACE_EMBEDDING_MODEL_PATH, 
    FACE_RECOGNITION_THRESHOLD,
    EMBEDDINGS_FILE,
    NAMES_FILE
)

# Load the face embedding model
try:
    print(f"Attempting to load model from {FACE_EMBEDDING_MODEL_PATH}")
    embedding_model = keras.models.load_model(FACE_EMBEDDING_MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    print(f"Full path tried: {FACE_EMBEDDING_MODEL_PATH}")
    print(f"Current working directory: {os.getcwd()}")
    embedding_model = None

def preprocess_face(face_image):
    """
    Preprocess a face image for the embedding model.
    
    Args:
        face_image: Input face image
        
    Returns:
        Preprocessed face image ready for embedding extraction
    """
    # Resize the image to the expected input size
    face_img = keras.preprocessing.image.smart_resize(face_image, (64, 64), "bicubic")
    
    # Preprocess input for the model
    face_img_processed = keras.applications.resnet.preprocess_input(face_img)
    
    return face_img_processed

def get_face_embedding(face_image):
    """
    Get the embedding vector for a face image.
    
    Args:
        face_image: Input face image
        
    Returns:
        Face embedding vector
    """
    if embedding_model is None:
        raise ValueError("Embedding model not loaded")
    
    preprocessed_face = preprocess_face(face_image)
    
    # Extract embedding
    embedding = embedding_model(np.expand_dims(preprocessed_face, axis=0)).numpy()[0]
    
    return embedding

def recognize_face(face_embedding):
    """
    Recognize a face by comparing its embedding to stored embeddings.
    
    Args:
        face_embedding: Face embedding vector
    
    Returns:
        Tuple of (is_recognized, name, distance) if recognized
        or (False, "", min_distance) if not recognized
    """
    try:
        # Load stored embeddings and names
        with open(EMBEDDINGS_FILE, 'rb') as f:
            embeddings = np.load(f)

        with open(NAMES_FILE, 'r') as f:
            names = f.read().splitlines()

    except FileNotFoundError:
        print("No stored embeddings or names found.")
        return False, "", float('inf')
    
    distances = np.array([distance(face_embedding, emb) for emb in embeddings])
    min_distance = np.min(distances)
    min_index = np.argmin(distances)
    min_name = names[min_index]
    
    print(f"Min distance: {min_distance}, Threshold: {FACE_RECOGNITION_THRESHOLD}")
    
    # Check if the face is recognized
    if min_distance < FACE_RECOGNITION_THRESHOLD:
        return True, min_name, min_distance
    
    return False, "", min_distance

def register_new_face(face_embedding, name):
    """
    Register a new face by storing its embedding and name.
    
    Args:
        face_embedding: Face embedding vector
        name: Name of the person
        
    Returns:
        Success status
    """
    
    # Check if the embedding file exists
    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            embeddings = np.load(f)
            embeddings = np.vstack([embeddings, face_embedding])

    except FileNotFoundError:
        embeddings = np.array([face_embedding])

    finally:
        # Save the embeddings and name
        with open(EMBEDDINGS_FILE, 'wb') as f:
            np.save(f, embeddings)
            
        # Append the name to the names file
        with open(NAMES_FILE, 'a') as f:
            f.write(name + '\n')
            
    return True
