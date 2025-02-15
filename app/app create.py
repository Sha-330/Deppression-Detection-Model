import os
import joblib
import cv2
import numpy as np
import json
import librosa
import logging
import tempfile
import moviepy.editor as mp
from flask import Flask, render_template, request, redirect, url_for, Response
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Define upload folders
UPLOAD_FOLDER = r'path to your folder that the images need to upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define paths to trained models
TEXT_MODEL_PATH = r'path to you model'
TEXT_VECTORIZER_PATH = r'.../text prediction/trained model/vectorizer.pkl'
AUDIO_MODEL_PATH = r'..../speech prediction/trained_model/decision_tree_model.joblib'
IMAGE_MODEL_PATH = r'.....facial prediction/trained model/decision_tree_model.joblib'
HAAR_CASCADE_PATH = r'...../facial prediction/haarcascade_frontalface_default.xml'

# Initialize VGG16 model for image and webcam predictions
vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Load text prediction model and vectorizer
text_clf = joblib.load(TEXT_MODEL_PATH)
text_vectorizer = joblib.load(TEXT_VECTORIZER_PATH)

# Load audio prediction model
audio_clf = joblib.load(AUDIO_MODEL_PATH)

# Load image prediction model and Haar Cascade for face detection
image_clf = joblib.load(IMAGE_MODEL_PATH)
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# Define emotion labels
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sadness', 5: 'surprise', 6: 'neutral'}
depressed_emotions = ['sadness', 'emptiness', 'worry', 'shame']
non_depressed_emotions = ['happiness', 'joy', 'relief', 'surprise', 'enthusiasm', 'love', 'neutral', 'anger', 'boredom', 'disgust', 'hate', 'fear']

# Home route
@app.route('/')
def home():
    return render_template('hero.html')

@app.route('/second')
def second():
    return render_template('second.html')

@app.route('/audio')
def audio():
    return render_template('audio.html')

@app.route('/image')
def image_route():
    return render_template('image.html')

@app.route('/text')
def text():
    return render_template('text.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

# Predict audio route
@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    if request.method == 'POST':
        try:
            if 'audiofile' not in request.files:
                return redirect(request.url)
            
            file = request.files['audiofile']
            if file.filename == '':
                return redirect(request.url)
            
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'audio', filename)
                file.save(filepath)
                
                depression_status = predict_depression(filepath)
                return render_template('audio.html', prediction=f'Predicted Depression Status: {depression_status}')
        
        except Exception as e:
            logging.error(f"Error predicting audio: {e}")
            return render_template('audio.html', prediction='Error predicting audio')

    return redirect(request.url)

# Predict image route
@app.route('/predict_image', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        try:
            if 'imagefile' not in request.files:
                return redirect(request.url)
            
            file = request.files['imagefile']
            if file.filename == '':
                return redirect(request.url)
            
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'image', filename)
                file.save(filepath)
                
                emotion, emotion_type = predict_emotion(filepath)
                return render_template('image.html', predicted_emotion=emotion, depression_status=emotion_type)
        
        except Exception as e:
            logging.error(f"Error predicting image: {e}")
            return render_template('image.html', predicted_emotion='Error predicting image', depression_status='')

    return redirect(request.url)

# Predict text route
@app.route('/predict_text', methods=['POST'])
def predict_text():
    if request.method == 'POST':
        try:
            text = request.form.get('text_input', '')
            result = predict_text_emotion(text)
            return render_template('text.html', result=result)
        
        except Exception as e:
            logging.error(f"Error predicting text emotion: {e}")
            return 'Error predicting text emotion'

    return redirect(request.url)

# Function to predict emotion from image
def predict_emotion(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        
        feature = vgg_model.predict(img_data)
        prediction = image_clf.predict(feature.flatten().reshape(1, -1))
        emotion_id = prediction[0]
        emotion = emotion_labels[emotion_id]

        emotion_type = "Depressed" if emotion_id in [2, 4] else "Non-depressed"
        
        return emotion, emotion_type
    
    except Exception as e:
        logging.error(f"Error processing image {img_path}: {e}")
        return None, None

# Function to predict depression status from audio
def predict_depression(file_path):
    depression_emotions = [
        'OAF_Sad', 'OAF_Fear', 'OAF_angry', 'OAF_disgust',
        'YAF_angry', 'YAF_disgust', 'YAF_fear', 'YAF_sad'
    ]
    
    try:
        # Extract audio features
        features = extract_audio_features(file_path, sample_rate=44100)
        if features is None:
            return None
        
        # Predict emotion
        features = features.reshape(1, -1)
        predicted_emotion = audio_clf.predict(features)[0]
        
        return "Depressed" if predicted_emotion in depression_emotions else "Not Depressed"
    
    except Exception as e:
        logging.error(f"Error predicting depression for {file_path}: {e}")
        return None

# Function to extract audio features
def extract_audio_features(file_path, sample_rate):
    try:
        _, file_extension = os.path.splitext(file_path)
        if file_extension == '.wav':
            audio_data, sr = librosa.load(file_path, sr=sample_rate)
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio_file:
                video = mp.VideoFileClip(file_path)
                video.audio.write_audiofile(temp_audio_file.name, codec='pcm_s16le')
                audio_data, sr = librosa.load(temp_audio_file.name, sr=sample_rate)

        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        return mfccs_mean
    
    except Exception as e:
        logging.error(f"Error extracting features from {file_path}: {e}")
        return None

# Function to predict emotion from text input
def predict_text_emotion(text):
    try:
        text_transformed = text_vectorizer.transform([text])
        predicted_emotion = text_clf.predict(text_transformed)[0]
        
        if predicted_emotion.lower() in depressed_emotions:
            emotion_type = "Depressed"
        elif predicted_emotion.lower() in non_depressed_emotions:
            emotion_type = "Non-depressed"
        else:
            emotion_type = "Unknown"
        
        return f"Predicted Emotion: {predicted_emotion} ({emotion_type})"
    
    except Exception as e:
        logging.error(f"Error predicting emotion for text: {e}")
        return "Error"

# Function to predict emotion from webcam frame
def predict_emotion_from_frame(face):
    try:
        img = cv2.resize(face, (224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        
        feature = vgg_model.predict(img_data)
        prediction = image_clf.predict_proba(feature.flatten().reshape(1, -1))[0]
        
        return prediction.tolist()
    except Exception as e:
        print(f"Error processing frame: {e}")
        return [0] * len(emotion_labels)

# Generator function to capture frames from webcam
def webcam_gen():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Error: Could not open webcam.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        prediction = [0] * len(emotion_labels)  # Initialize prediction
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            try:
                prediction = predict_emotion_from_frame(face)
            except Exception as e:
                print(f"Error processing frame: {e}")
                prediction = [0] * len(emotion_labels)  # Set a default value if there's an error
            
            emotion_id = np.argmax(prediction)
            emotion_label = emotion_labels[emotion_id]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        
        prediction_data = json.dumps(prediction)
        socketio.emit('emotion_prediction', prediction_data, namespace='/socket')

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(webcam_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect', namespace='/socket')
def connect():
    print('Client connected')

if __name__ == '__main__':
    socketio.run(app, debug=True)
