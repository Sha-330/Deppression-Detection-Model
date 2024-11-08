import os
import numpy as np
import librosa
import joblib
import logging
import moviepy.editor as mp
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
save_dir = r'D:/depression/app/trained_model'
model_name = 'decision_tree_model.joblib'
model_path = os.path.join(save_dir, model_name)
dtree = joblib.load(model_path)

# Define emotions indicating depression
depression_emotions = [
    'OAF_Sad', 'OAF_Fear', 'OAF_angry', 'OAF_disgust',
    'YAF_angry', 'YAF_disgust', 'YAF_fear', 'YAF_sad'
]

# Function to extract MFCC features from audio or video data
def extract_audio_features(file_path, sample_rate):
    try:
        _, file_extension = os.path.splitext(file_path)
        if file_extension == '.wav':
            audio_data, sr = librosa.load(file_path, sr=sample_rate)
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio_file:
                # Convert video to audio
                video = mp.VideoFileClip(file_path)
                video.audio.write_audiofile(temp_audio_file.name, codec='pcm_s16le')

                # Load converted audio
                audio_data, sr = librosa.load(temp_audio_file.name, sr=sample_rate)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean

    except Exception as e:
        logging.error(f"Error extracting features from {file_path}: {e}")
        return None

# Function to predict depression status from audio or video file
def predict_depression(file_path, sample_rate=44100):
    try:
        # Extract features from audio (or video)
        features = extract_audio_features(file_path, sample_rate)
        if features is None:
            return None
        
        # Reshape features for prediction (sklearn expects 2D array)
        features = features.reshape(1, -1)
        
        # Predict using the loaded model
        predicted_emotion = dtree.predict(features)[0]
        
        # Check if predicted emotion indicates depression
        if predicted_emotion in depression_emotions:
            return "Depressed"
        else:
            return "Not Depressed"
    
    except Exception as e:
        logging.error(f"Error predicting depression for {file_path}: {e}")
        return None

# Example usage:
if __name__ == "__main__":
    # Replace with the path to your test audio or video file
    test_file = r'D:/depression/app/static/uploads/audio/example_audio.wav'
    
    # Predict depression status
    predicted_depression_status = predict_depression(test_file)
    
    if predicted_depression_status is not None:
        print(f"Predicted Depression Status: {predicted_depression_status}")
