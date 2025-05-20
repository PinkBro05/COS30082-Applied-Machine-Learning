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

def eye_aspect_ratio(eye):
    """
    Calculate the eye aspect ratio for blink detection.
    
    Based on research from pyimagesearch
    (https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/)
    
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
