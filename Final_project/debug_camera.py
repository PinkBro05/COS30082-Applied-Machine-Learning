#!/usr/bin/env python3
"""
Advanced camera debugging utility for macOS.
This script provides detailed information about available cameras and attempts multiple methods
to access the camera to help diagnose issues.
"""

import cv2
import time
import platform
import sys
import numpy as np

def print_camera_backends():
    """Print all available camera backends in OpenCV"""
    backends = [
        (cv2.CAP_ANY, "CAP_ANY: Auto-detect"),
        (cv2.CAP_DSHOW, "CAP_DSHOW: DirectShow (Windows)"),
        (cv2.CAP_MSMF, "CAP_MSMF: Media Foundation (Windows)"),
        (cv2.CAP_V4L2, "CAP_V4L2: Video for Linux"),
        (cv2.CAP_AVFOUNDATION, "CAP_AVFOUNDATION: AVFoundation (macOS)"),
        (cv2.CAP_IMAGES, "CAP_IMAGES: Image sequence"),
        (cv2.CAP_FFMPEG, "CAP_FFMPEG: FFMPEG"),
        (cv2.CAP_QT, "CAP_QT: QuickTime")
    ]
    
    print("Available camera backends in OpenCV:")
    for code, name in backends:
        print(f"- {name} (code: {code})")

def test_camera_with_backend(backend_id, backend_name):
    """Test camera with a specific backend"""
    print(f"\nTesting camera with {backend_name} backend...")
    
    try:
        cap = cv2.VideoCapture(0, backend_id)
        time.sleep(1.0)
        
        if not cap.isOpened():
            print(f"❌ Failed to open camera with {backend_name} backend")
            cap.release()
            return False, None
        
        # Read a test frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"❌ Camera opened but failed to grab frame with {backend_name} backend")
            cap.release()
            return False, None
        
        # Display camera properties
        print(f"✅ Successfully accessed camera with {backend_name} backend")
        print(f"   Frame dimensions: {frame.shape}")
        print(f"   Frame type: {frame.dtype}")
        print(f"   Frame width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"   Frame height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"   FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        
        # Write a sample frame to help with debugging
        test_filename = f"camera_test_{backend_name.replace(' ', '_').replace(':', '')}.jpg"
        try:
            cv2.imwrite(test_filename, frame)
            print(f"   Sample frame saved to {test_filename}")
        except Exception as e:
            print(f"   Could not save test image: {e}")
        
        # Release the camera
        cap.release()
        return True, frame
        
    except Exception as e:
        print(f"❌ Error testing {backend_name} backend: {e}")
        return False, None

def check_frame_compatibility(frame):
    """Check if the frame is compatible with face recognition processing"""
    if frame is None:
        print("❌ Frame is None, not compatible for processing")
        return False
    
    print("\nFrame compatibility check:")
    
    # Check dimensions
    h, w = frame.shape[:2]
    print(f"- Dimensions: {w}x{h} pixels")
    if w < 64 or h < 64:
        print("❌ Frame is too small (minimum 64x64 recommended)")
        return False
    
    # Check channel count
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        print("❌ Frame is not a 3-channel RGB image")
        return False
    else:
        print("✅ Frame has 3 channels (RGB)")
    
    # Check data type
    print(f"- Data type: {frame.dtype}")
    if frame.dtype != np.uint8:
        print("❌ Frame is not 8-bit unsigned integer format")
        return False
    else:
        print("✅ Frame is 8-bit format")
    
    # Check if frame contains actual data (not all zeros or ones)
    min_val = frame.min()
    max_val = frame.max()
    mean_val = frame.mean()
    print(f"- Pixel value range: min={min_val}, max={max_val}, mean={mean_val:.2f}")
    
    if min_val == max_val:
        print("❌ Frame has uniform values (might be empty or corrupted)")
        return False
    
    print("✅ Frame appears valid for processing")
    return True

def main():
    """Main function to test camera access"""
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    
    print_camera_backends()
    
    # Test different backends
    test_backends = [
        (cv2.CAP_ANY, "Auto-detect"),
        (cv2.CAP_AVFOUNDATION, "AVFoundation (macOS)"),
        (cv2.CAP_QT, "QuickTime")
    ]
    
    # For non-macOS, adjust backends
    if platform.system() != "Darwin":
        if platform.system() == "Windows":
            test_backends = [
                (cv2.CAP_ANY, "Auto-detect"),
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "Media Foundation")
            ]
        elif platform.system() == "Linux":
            test_backends = [
                (cv2.CAP_ANY, "Auto-detect"),
                (cv2.CAP_V4L2, "V4L2")
            ]
    
    successful_frame = None
    for backend_id, backend_name in test_backends:
        success, frame = test_camera_with_backend(backend_id, backend_name)
        if success and frame is not None:
            if successful_frame is None:
                successful_frame = frame
    
    # Check frame compatibility if we got a successful frame
    if successful_frame is not None:
        check_frame_compatibility(successful_frame)
        
    print("\nCamera Test Summary:")
    
    if successful_frame is not None:
        print("✅ At least one backend successfully accessed the camera!")
        print("\nRecommendation:")
        print("When initializing your camera in the app, use:")
        
        # Find the first successful backend
        for backend_id, backend_name in test_backends:
            success, _ = test_camera_with_backend(backend_id, backend_name)
            if success:
                print(f"cap = cv2.VideoCapture(0, {backend_id})  # {backend_name}")
                break
    else:
        print("❌ Failed to access the camera with any backend")
        print("\nPlease check the following:")
        print("1. Make sure your camera is properly connected")
        print("2. Check camera permissions in System Preferences > Security & Privacy > Privacy > Camera")
        print("3. Close other applications that might be using the camera")
        print("4. Restart your computer")
        
if __name__ == "__main__":
    main()
