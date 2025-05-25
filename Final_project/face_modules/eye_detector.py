"""
Eye detection and analysis module.
Provides functionality for eye blink detection and other eye-related features.
"""

import numpy as np
from face_modules.face_detector import landmark_predictor
from face_modules.utils.math_utils import eye_aspect_ratio
from face_modules.config import LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS, EYE_ASPECT_RATIO_THRESHOLD

def get_eye_landmarks(image, face_detection):
    """
    Get eye landmarks for a detected face.
    
    Args:
        image: Input image
        face_detection: Face detection from dlib
        
    Returns:
        Tuple of (left_eye_landmarks, right_eye_landmarks)
    """
    # Get the facial landmarks
    landmarks = landmark_predictor(image, face_detection)
    
    # Extract the eye landmark coordinates
    left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_LANDMARKS])
    right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_LANDMARKS])
    
    return left_eye, right_eye

def check_eye_status(image, face_detection):
    """
    Check if eyes are open or closed based on eye aspect ratio.
    
    Args:
        image: Input image
        face_detection: Face detection from dlib
        
    Returns:
        Tuple of (eye_state, eye_aspect_ratio)
        eye_state is either "eyes_opened" or "eyes_closed"
    """
    # Get eye landmarks
    left_eye, right_eye = get_eye_landmarks(image, face_detection)
    
    # Calculate eye aspect ratios
    left_eye_aspect_ratio = eye_aspect_ratio(left_eye)
    right_eye_aspect_ratio = eye_aspect_ratio(right_eye)
    
    # Average of both eyes
    ear = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0
    
    # Determine if eyes are open or closed
    if ear > EYE_ASPECT_RATIO_THRESHOLD:
        return "eyes_opened", ear
    else:
        return "eyes_closed", ear

