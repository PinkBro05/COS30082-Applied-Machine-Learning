import face_recognition
import cv2
import numpy as np

# Load a sample picture and learn how to recognize it.
hcm_image = face_recognition.load_image_file("Week_8/HoChiMinh.jfif")
pink_image = face_recognition.load_image_file("Week_8/pink.jfif")
faker_face_encoding = face_recognition.load_image_file("Week_8/Faker.jpg")

# Get the face encodings for the first face in the image.
hcm_face_encoding = face_recognition.face_encodings(hcm_image)[0]
pink_face_encoding = face_recognition.face_encodings(pink_image)[0]
faker_face_encoding = face_recognition.face_encodings(faker_face_encoding)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    hcm_face_encoding,
    pink_face_encoding,
    faker_face_encoding
]
known_face_names = [
    "Ho Chi Minh",
    "Pink",
    "Faker"
]

# Detect faces in a video stream
video_capture = cv2.VideoCapture(0)
while True:
    # Capture a single frame from the video stream
    ret, frame = video_capture.read()

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
        # Check if the face is a match for any known faces
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        else:
            # If no match is found, use the best match index to get the name
            name = known_face_names[best_match_index] if best_match_index < len(known_face_names) else "Unknown"
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break