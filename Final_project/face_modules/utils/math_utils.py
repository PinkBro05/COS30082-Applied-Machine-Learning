"""
Utility functions for face recognition system.
"""

import numpy as np

def distance(emb1, emb2):
    """
    Calculate the Euclidean distance between two embeddings.
    
    Args:
        emb1: First embedding vector
        emb2: Second embedding vector
        
    Returns:
        The squared Euclidean distance between the embeddings
    """
    return np.sum(np.square(emb1 - emb2))

def normalize_embedding(embedding):
    """
    Normalize an embedding vector to unit length.
    
    Args:
        embedding: The embedding vector to normalize
        
    Returns:
        Normalized embedding vector with unit length
    """
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding

def eye_aspect_ratio(eye):
    """
    Calculate the eye aspect ratio for blink detection.
    
    Args:
        eye: Array of eye landmark coordinates
        
    Returns:
        The eye aspect ratio value
    """
    # euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear
