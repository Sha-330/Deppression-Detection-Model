import cv2
import joblib
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import time

# Load the trained model
model_path = "..../facial prediction/trained model/decision_tree_model.joblib"
clf = joblib.load(model_path)

# Define emotions and corresponding labels
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

# Define depressed and non-depressed emotions with their respective labels
depressed_emotions = {4: 'sadness', 2: 'fear'}
non_depressed_emotions = {0: 'angry', 1: 'disgust', 3: 'happy', 5: 'surprise', 6: 'neutral'}

# Initialize the VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier("D:/depression/facial prediction/haarcascade_frontalface_default.xml")

# Function to predict emotion from image data
def predict_emotion_from_frame(face):
    try:
        img = cv2.resize(face, (224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        
        feature = vgg_model.predict(img_data)
        prediction = clf.predict(feature.flatten().reshape(1, -1))
        emotion_id = prediction[0]
        emotion = emotion_labels[emotion_id]
        
        # Determine if the emotion is depressed or non-depressed
        if emotion_id in depressed_emotions:
            emotion_type = "Depressed"
        else:
            emotion_type = "Non-depressed"
        
        return emotion, emotion_type
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None, None

# Start webcam and live prediction
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]
        
        # Predict emotion from the face region
        emotion, emotion_type = predict_emotion_from_frame(face)
        
        # Draw a rectangle around the face and display the emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if emotion:
            text = f"{emotion} ({emotion_type})"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Show the frame with the detected faces and predicted emotions
    cv2.imshow('Depression Prediction', frame)
    
    # Add delay to control frame rate (optional, e.g., 30 FPS)
    time.sleep(1/30)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
