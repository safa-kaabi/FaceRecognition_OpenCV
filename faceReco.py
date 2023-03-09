import cv2
import face_recognition
import os
import numpy as np

# Path to the directory containing the known images
known_path = 'C:/Users/Safa/Desktop/FaceDetection/Khnow'

# Path to the directory containing the unknown images
unknown_path = 'C:/Users/Safa/Desktop/FaceDetection/Unknown'

# Load the known face encodings
known_encodings = []
known_names = []
for filename in os.listdir(known_path):
    image = face_recognition.load_image_file(os.path.join(known_path, filename))
    encoding = face_recognition.face_encodings(image)[0]
    known_encodings.append(encoding)
    known_names.append(os.path.splitext(filename)[0])

# Load the unknown images and perform face detection
for filename in os.listdir(unknown_path):
    image = cv2.imread(os.path.join(unknown_path, filename))
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
# Start the video capture
video_capture = cv2.VideoCapture(0)
while True:
    # Capture each frame from the video
    ret, frame = video_capture.read()

    # Convert the frame from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Perform Haar Cascade face detection on the RGB frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Convert the face coordinates to the format used by face_recognition
    face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]

    # Encode the faces using face_recognition
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Iterate through each face in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face with the known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        # Find the best match for the face
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        # If there is a match, set the name to the known name
        if matches[best_match_index]:
            name = known_names[best_match_index]

        # Draw a box around the face and label it with the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit the program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
video_capture.release()
cv2.destroyAllWindows()


