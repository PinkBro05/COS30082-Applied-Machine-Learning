"""
Face-based check-in system module.
Integrates face detection, eye analysis, and face recognition for a complete check-in system.
"""

import numpy as np
from face_modules.face_detector import detect_faces, get_face_bbox
from face_modules.eye_detector import check_eye_status
from face_modules.face_recognition import get_face_embedding, recognize_face, register_new_face

def process_check_in(image, state):
    """
    Process a check-in request with face recognition and liveness detection via eye blinking.
    
    Args:
        image: Input camera image
        state: Current state dictionary with:
            - last_face: Name of the last recognized face
            - taken_actions: Set of recorded eye actions (opened/closed)
            
    Returns:
        Tuple of (eye_result, recognition_result, check_in_result, annotated_image, updated_state)
    """
    if image is None:
        # Create a blank image instead of None to avoid AnnotatedImage error
        blank_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        return "", "Press recording before checkin", "", (blank_image, []), state

    # Detect faces in the image
    face_detections = detect_faces(image, upsample_num_times=2)
    
    # Check the number of faces detected
    if len(face_detections) > 1:
        state["taken_actions"] = set()
        return "", "Multiple faces detected. Please try one face at a time.", "", (image, []), state
    
    elif len(face_detections) == 0:
        state["taken_actions"] = set()
        return "", "No face detected. Please try again.", "", (image, []), state
    
    # Get the detected face
    face_det = face_detections[0]
    
    # Check eye state (open/closed)
    eye_action, ear = check_eye_status(image, face_det)
    
    # Get face bounding box
    bbox, face_boundary = get_face_bbox(image, face_det)
    
    # Crop face from image
    face_img = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    
    # Get face embedding
    face_embedding = get_face_embedding(face_img)
    
    # Recognize face
    registered, face_min_name, face_min_distance = recognize_face(face_embedding)
    
    if not registered:
        state["taken_actions"] = set()
        return (
            f"{eye_action}. Ratio: {ear}",
            f"Face not registered. Min distance: {face_min_distance}",
            "",
            (image, [(face_boundary, "Unregistered face")]),
            state
        )
    
    # Handle recognized face
    if face_min_name == state["last_face"]:
        state["taken_actions"].add(eye_action)
        # Check if both eye states have been observed (blink detection)
        if len(state["taken_actions"]) == 2:
            # Anti-spoofing passed - show the name
            return (
                f"{eye_action}. Ratio: {ear}",
                f"Recognized {face_min_name}. Distance: {face_min_distance}",
                f"{face_min_name} checked-in",
                (image, [(face_boundary, f"{face_min_name}")]),
                state
            )
        else:
            # Anti-spoofing not yet passed - don't show the name
            return (
                f"{eye_action}. Ratio: {ear}",
                f"Person detected. Distance: {face_min_distance}",
                f"Waiting for eye action",
                (image, [(face_boundary, f"Verification required")]),
                state
            )
    else:        # New face detected, reset state
        state["taken_actions"] = {eye_action}
        state["last_face"] = face_min_name
        return (
            f"{eye_action}. Ratio: {ear}",
            f"Person detected. Distance: {face_min_distance}",
            f"Waiting for eye action",
            (image, [(face_boundary, f"Verification required")]),
            state
        )

def process_registration(image, name):
    """
    Register a new face in the system.
    
    Args:
        image: Input camera image
        name: Name to associate with the face
        
    Returns:
        Tuple of (registration_result, annotated_image)
    """
    
    # Detect faces in the image
    face_detections = detect_faces(image, upsample_num_times=1)
    
    # Check the number of faces detected
    if len(face_detections) > 1:
        return (
            "Multiple faces detected. Please register one face at a time.",
            (image, [((det.left(), det.top(), det.right(), det.bottom()), f"Face #{i}") 
                     for i, det in enumerate(face_detections)])
        )
    
    elif len(face_detections) == 0:
        return "No face detected. Please try again.", (image, [])
    
    # Get the detected face
    face_det = face_detections[0]
    
    # Get face bounding box
    bbox, _ = get_face_bbox(image, face_det)
    
    # Create sections for annotation
    sections = [((bbox[0], bbox[1], bbox[2], bbox[3]), name)]
    
    # Crop face from image
    face_img = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    
    # Get face embedding
    face_embedding = get_face_embedding(face_img)
    
    # Register the new face
    register_new_face(face_embedding, name)
    
    return f"Registered one face for {name}", (image, sections)
