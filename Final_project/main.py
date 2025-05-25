"""
Main application for face recognition check-in system.
Uses the modular components from the face_modules package.
"""

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from face_modules.check_in_system import process_check_in, process_registration
import threading
import time
import platform

class FaceRecognitionApp:
    def __init__(self, root):
        """Initialize the Face Recognition GUI application"""
        self.root = root
        self.root.title("Face Recognition Check-In System")
        self.root.geometry("1000x600")
        self.root.resizable(False, False)
        
        # State variables
        self.frame = None
        self.cap = None
        self.thread_running = False
        self.app_state = {
            "last_face": "",
            "taken_actions": set()
        }
        self.mode = "check_in"  # Default mode: check_in or register
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Camera feed
        self.camera_frame = ttk.LabelFrame(main_frame, text="Camera Feed")
        self.camera_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Camera canvas
        self.camera_canvas = tk.Canvas(self.camera_frame, width=640, height=480, bg="black")
        self.camera_canvas.pack(padx=10, pady=10)
        
        # Right panel - Controls and status
        controls_frame = ttk.LabelFrame(main_frame, text="Controls")
        controls_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Camera controls
        camera_controls = ttk.Frame(controls_frame)
        camera_controls.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_button = ttk.Button(camera_controls, text="Start Camera", command=self.start_camera)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.stop_button = ttk.Button(camera_controls, text="Stop Camera", command=self.stop_camera)
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)
        self.stop_button.config(state=tk.DISABLED)
        
        # Mode selection
        mode_frame = ttk.LabelFrame(controls_frame, text="Mode")
        mode_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.mode_var = tk.StringVar(value="check_in")
        ttk.Radiobutton(mode_frame, text="Check-In", variable=self.mode_var, 
                         value="check_in", command=self.mode_changed).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(mode_frame, text="Registration", variable=self.mode_var, 
                         value="register", command=self.mode_changed).pack(anchor=tk.W, padx=5, pady=2)
        
        # Registration section
        self.register_frame = ttk.LabelFrame(controls_frame, text="Registration")
        self.register_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(self.register_frame, text="Name:").pack(anchor=tk.W, padx=5, pady=2)
        self.name_entry = ttk.Entry(self.register_frame, width=20)
        self.name_entry.pack(fill=tk.X, padx=5, pady=2)
        
        self.register_button = ttk.Button(self.register_frame, text="Register Face", 
                                          command=self.register_face)
        self.register_button.pack(padx=5, pady=5)
        
        # Status section
        status_frame = ttk.LabelFrame(controls_frame, text="Status")
        status_frame.pack(fill=tk.X, padx=10, pady=10, expand=True)
        
        ttk.Label(status_frame, text="Eye Status:").pack(anchor=tk.W, padx=5, pady=2)
        self.eye_status_label = ttk.Label(status_frame, text="-")
        self.eye_status_label.pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Label(status_frame, text="Recognition:").pack(anchor=tk.W, padx=5, pady=2)
        self.recognition_label = ttk.Label(status_frame, text="-")
        self.recognition_label.pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Label(status_frame, text="Check-In Result:").pack(anchor=tk.W, padx=5, pady=2)
        self.checkin_result_label = ttk.Label(status_frame, text="-")
        self.checkin_result_label.pack(anchor=tk.W, padx=5, pady=2)
        
        # Set layout weights
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Initially hide registration controls
        self.mode_changed()
        
    def mode_changed(self):
        """Handle mode change between check-in and registration"""
        self.mode = self.mode_var.get()
        if self.mode == "check_in":
            self.register_frame.pack_forget()
        else:
            self.register_frame.pack(fill=tk.X, padx=10, pady=10)
    
    def start_camera(self):
        """Start camera capture and processing"""
        try:
            # Show permission instructions for macOS
            if platform.system() == 'Darwin':  # macOS
                messagebox.showinfo(
                    "Camera Permission", 
                    "This app needs camera permission to function.\n\n"
                    "If you see a permission dialog, please click 'OK'.\n\n"
                    "If the camera still doesn't work, please go to:\n"
                    "System Preferences > Security & Privacy > Privacy > Camera\n"
                    "and make sure Python/Terminal has permission."
                )
            
            # Explicitly close any previously opened camera
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
                self.cap = None
                time.sleep(0.5)
            
            # Try capturing video with different backends
            print("Attempting to open camera...")
            
            # Try multiple backends on macOS
            if platform.system() == 'Darwin':
                try_backends = [cv2.CAP_ANY, cv2.CAP_AVFOUNDATION, cv2.CAP_QT]
            else:
                try_backends = [cv2.CAP_ANY]
                
            camera_opened = False
            for backend in try_backends:
                if camera_opened:
                    break
                    
                print(f"Trying camera with backend {backend}...")
                self.cap = cv2.VideoCapture(1, backend)
                time.sleep(1.0)  # Wait for camera initialization
            
            # Check if camera opened successfully
            if not self.cap.isOpened():
                # Try an alternative method on macOS
                if platform.system() == 'Darwin':
                    # Close and reopen the camera with specific parameters for macOS
                    self.cap.release()
                    self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION) # Change based on the system
                    time.sleep(0.5)
                    
                # If still not working
                if not self.cap.isOpened():
                    raise ValueError("Could not open webcam. Please check camera permissions.")
                
            # Read a test frame to verify camera is working
            ret, test_frame = self.cap.read()
            if not ret:
                raise ValueError("Camera opened but failed to grab frame.")
                
            self.thread_running = True
            self.camera_thread = threading.Thread(target=self.camera_stream)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            # Update button states
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
        except Exception as e:
            self.show_error(f"Error starting camera: {e}")
            if platform.system() == 'Darwin':
                messagebox.showinfo(
                    "Camera Troubleshooting", 
                    "Please try the following:\n\n"
                    "1. Close other applications using the camera\n"
                    "2. Check camera permissions in System Preferences\n"
                    "3. Restart your computer\n\n"
                    "Terminal command to check camera access:\n"
                    "tccutil query camera"
                )
    
    def stop_camera(self):
        """Stop camera capture and processing"""
        self.thread_running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=1)
            
        if self.cap and self.cap.isOpened():
            self.cap.release()
            
        # Update button states
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        # Clear image canvas
        self.camera_canvas.delete("all")
        self.camera_canvas.create_text(320, 240, text="Camera Off", fill="white", font=("Arial", 20))
    
    def camera_stream(self):
        """Process camera stream in a separate thread"""
        try:
            while self.thread_running:
                # Grab a frame from the camera
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.root.after(0, self.show_error, "Failed to grab frame from camera")
                    self.root.after(0, self.stop_camera)
                    break
                
                # Make sure the frame is valid
                if frame.size == 0:
                    print("Warning: Empty frame received, skipping...")
                    time.sleep(0.033)
                    continue
                
                # Validate frame format
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    print(f"Warning: Unexpected frame format: {frame.shape}, converting...")
                    # Try to convert to RGB if it's not already
                    if len(frame.shape) == 2:  # Grayscale
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.shape[2] == 4:  # RGBA
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                
                # Store the processed frame
                self.frame = cv2.flip(frame, 1)  # Flip horizontally for a more natural view
                
                # Process the frame based on current mode
                sections = []
                if self.mode == "check_in":
                    try:
                        # Process check-in
                        eye_result, recognition_result, check_in_result, (annotated_frame, sections), self.app_state = (
                            process_check_in(self.frame, self.app_state)
                        )
                        
                        # Update GUI labels
                        self.root.after(0, self.update_status_labels, eye_result, recognition_result, check_in_result)
                    except Exception as e:
                        print(f"Error in check-in processing: {e}")
                        # Just display the frame without processing
                        self.root.after(0, self.update_status_labels, f"Error: {str(e)}", "Processing failed", "")
                
                # Display the frame (for both check-in and registration modes)
                self.root.after(0, self.display_frame, self.frame, sections)
                
                # Sleep to control frame rate
                # time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            self.root.after(0, self.show_error, f"Camera error: {str(e)}")
            self.root.after(0, self.stop_camera)
    
    def display_frame(self, frame, sections):
        """Display the frame with annotations on the canvas"""
        try:
            # Draw rectangles for detected faces
            frame_copy = frame.copy()  # Create a copy to avoid modifying the original frame
            for rect, label in sections:
                x1, y1, x2, y2 = rect
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_copy, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert to PIL format for Tkinter
            rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            self.tk_image = ImageTk.PhotoImage(image=pil_image)
            
            # Update canvas
            self.camera_canvas.delete("all")  # Clear previous image
            self.camera_canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)
        except Exception as e:
            # Handle display errors gracefully
            self.camera_canvas.delete("all")
            self.camera_canvas.create_text(320, 240, text=f"Display Error: {str(e)}", fill="red", font=("Arial", 14))
            print(f"Error displaying frame: {e}")
    
    def update_status_labels(self, eye_result, recognition_result, check_in_result):
        """Update the status labels with results"""
        self.eye_status_label.config(text=eye_result)
        self.recognition_label.config(text=recognition_result)
        self.checkin_result_label.config(text=check_in_result)
    
    def register_face(self):
        """Register a new face"""
        name = self.name_entry.get().strip()
        if not name:
            self.show_error("Please enter a name for registration")
            return
            
        if self.frame is None:
            self.show_error("Camera not started, please start the camera first")
            return
            
        try:
            # Process registration
            result, (annotated_frame, sections) = process_registration(self.frame, name)
            
            # Show result
            self.checkin_result_label.config(text=result, foreground="green")
            
            # Display the annotated frame with the registered face
            if sections and self.frame is not None:
                self.display_frame(self.frame, sections)
                
        except Exception as e:
            self.show_error(f"Registration failed: {str(e)}")
    
    def show_error(self, message):
        """Display an error message"""
        print(f"Error: {message}")
        
        # Update UI from main thread
        def update_ui():
            self.checkin_result_label.config(text=f"Error: {message}", foreground="red")
            
            # For critical errors, show a message box
            if "camera" in message.lower() or "webcam" in message.lower():
                messagebox.showerror("Camera Error", 
                                   f"{message}\n\nPlease check that:\n"
                                   "1. No other application is using the camera\n"
                                   "2. Camera permissions are granted\n"
                                   "3. Your camera is properly connected")
        
        # If called from a non-main thread, use after to schedule UI update on main thread
        if threading.current_thread() is not threading.main_thread():
            self.root.after(0, update_ui)
        else:
            update_ui()

def main():
    """Main entry point of the application"""
    # Create DB directory if it doesn't exist
    os.makedirs('db', exist_ok=True)
    
    # Check if required model files exist
    model_dir = 'model_factory'
    landmark_file = os.path.join(model_dir, 'shape_predictor_68_face_landmarks.dat')
    embedding_file = os.path.join(model_dir, 'new_models' ,'embedding_euclidean.keras')
    
    missing_files = []
    if not os.path.isfile(landmark_file):
        missing_files.append(landmark_file)
    if not os.path.isfile(embedding_file):
        missing_files.append(embedding_file)
        
    if missing_files:
        print("ERROR: The following required model files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure these files are in the model_factory directory before running the application.")
        print("You can download the landmark predictor from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        sys.exit(1)
    
    # Create Tkinter root window
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    
    # Print initial help message
    print("Face Recognition Check-In System")
    print("---------------------------------")
    print("1. Click 'Start Camera' to begin")
    print("2. Use the mode selection to switch between Check-In and Registration")
    print("3. For registration, enter a name and click 'Register Face'")
    print("4. For check-in, simply look at the camera and blink")
    print("---------------------------------\n")
    
    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()