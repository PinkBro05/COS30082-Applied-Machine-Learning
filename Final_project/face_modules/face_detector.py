"""
Face detection module.
Provides functionality for detecting faces in images using dlib.
"""

import dlib
import numpy as np
from face_modules.config import LANDMARK_PREDICTOR_PATH

# Initialize the face detector and landmark predictor once
detector = dlib.get_frontal_face_detector()  # type: ignore
try:
    landmark_predictor = dlib.shape_predictor(LANDMARK_PREDICTOR_PATH)  # type: ignore
except RuntimeError:
    print(f"Error: Could not load face landmark predictor from {LANDMARK_PREDICTOR_PATH}.")
    print("Make sure the shape_predictor_68_face_landmarks.dat file is in the correct location.")
    print("You can download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    landmark_predictor = None

def detect_faces(image, upsample_num_times=1):
    """
    Detect faces in an image.
    
    Args:
        image: Input image (numpy array)
        upsample_num_times: Number of times to upsample the image (increasing this can help detect smaller faces)
        
    Returns:
        List of detected face rectangles
    """
    # Detect faces in the image
    return detector(image, upsample_num_times)

def get_face_bbox(image, face_detection):
    """
    Get a square bounding box from a face detection.
    Makes sure the box is square and fits within the image boundaries.
    
    Args:
        image: Original image
        face_detection: Face detection rectangle from dlib
        
    Returns:
        Tuple of (left, top, right, bottom) coordinates for cropping
        and (left, top, right, bottom) as face_boundary for display
    """
    # Calculate the size as the max of width and height to ensure a square
    det_size = min(max(face_detection.width(), face_detection.height()), 
                   image.shape[0] // 2, image.shape[1] // 2)

    # Calculate the center of the face
    center_x = min(max(face_detection.center().x, det_size), image.shape[1] - det_size)
    center_y = min(max(face_detection.center().y, det_size), image.shape[0] - det_size)

    # Calculate the bounding box coordinates
    top = center_y - det_size
    bottom = center_y + det_size
    left = center_x - det_size
    right = center_x + det_size

    # Face boundary for display
    face_boundary = (left, top, right, bottom)
    
    return (left, top, right, bottom), face_boundary

def crop_face(image, bbox):
    """
    Crop a face from an image using the bounding box.
    
    Args:
        image: Input image
        bbox: Bounding box as (left, top, right, bottom)
        
    Returns:
        Cropped face image
    """
    left, top, right, bottom = bbox
    return image[top:bottom, left:right]
