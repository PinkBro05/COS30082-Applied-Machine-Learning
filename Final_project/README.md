# Face Recognition Check-In System

This application is a face recognition-based check-in system that uses computer vision techniques for face detection, eye blink detection for liveness verification, and face recognition.

## Features

- Real-time face detection
- Eye blink detection as a liveness check (anti-spoofing measure)
- Face recognition with embeddings
- User registration system
- GUI-based interface

## Requirements

- Python 3.8+
- OpenCV
- dlib
- TensorFlow/Keras
- PIL (Pillow)
- NumPy
- tkinter (included in standard Python)

## Setup

1. Make sure all required packages are installed:

```bash
pip install opencv-python dlib tensorflow pillow numpy
```

2. Download the required models:
   - shape_predictor_68_face_landmarks.dat (for facial landmarks detection)
   - embedding_euclidean.keras (for face embedding)
   
   These should be placed in the `model_factory` directory.

3. Run the application:

```bash
python main.py
```

## Usage

### Check-in Mode

1. Start the camera using the "Start Camera" button
2. The system will detect your face and attempt to recognize you
3. To complete check-in, blink your eyes (the system verifies liveness by detecting both open and closed eyes)
4. Once verified, your check-in will be confirmed

### Registration Mode

1. Select "Registration" mode
2. Enter your name in the input field
3. Start the camera if not already running
4. Position your face in front of the camera
5. Click "Register Face" button
6. Your face will be registered in the system

## Project Structure

- `main.py`: GUI application entry point
- `face_modules/`: Core modules for face processing
  - `face_detector.py`: Face detection using dlib
  - `eye_detector.py`: Eye analysis and blink detection
  - `face_recognition.py`: Face recognition and embedding extraction
  - `check_in_system.py`: Integration of all modules for the check-in workflow
  - `config.py`: Configuration settings
  - `utils/`: Utility functions
    - `math_utils.py`: Mathematical functions for face analysis
- `db/`: Storage for face embeddings and names
- `model_factory/`: Pre-trained models

## Notes

- For optimal face detection, ensure good lighting conditions
- Keep your face within frame during check-in and registration
- The system requires both open and closed eye states to confirm liveness
