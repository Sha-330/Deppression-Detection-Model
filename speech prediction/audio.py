import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Function to extract MFCC features from audio data
def extract_audio_features(audio_file, sample_rate):
    # Load audio file
    audio_data, sr = librosa.load(audio_file, sr=sample_rate)
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Directory where your dataset is stored
dataset_dir = r'..../speech/TESS Toronto emotional speech set data'

# Lists to store features and labels
X_audio = []
y = []

# Emotions in the TESS dataset
emotions = ['OAF_Fear', 'OAF_Pleasant_surprise', 'OAF_Sad', 'OAF_angry', 
            'OAF_disgust', 'OAF_happy', 'OAF_neutral', 'YAF_angry', 
            'YAF_disgust', 'YAF_fear', 'YAF_happy', 'YAF_neutral', 
            'YAF_pleasant_surprised', 'YAF_sad']

# Iterate through each emotion folder
for emotion in emotions:
    # Path to emotion folder
    emotion_folder = os.path.join(dataset_dir, emotion)
    # Iterate through each audio file (WAV format)
    audio_files = os.listdir(emotion_folder)
    for audio_file in audio_files:
        # Load audio file path
        audio_path = os.path.join(emotion_folder, audio_file)
        
        # Extract audio features (MFCCs)
        mfccs = extract_audio_features(audio_path, sample_rate=44100)  # Adjust sample_rate if necessary

        # Append features and labels to lists
        X_audio.append(mfccs)
        y.append(emotion)  # Use the folder name (emotion) as label

# Convert lists to numpy arrays
X_audio = np.asarray(X_audio)
y = np.asarray(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_audio, y, test_size=0.2, random_state=42)

# Initialize Decision Tree classifier
dtree = DecisionTreeClassifier(random_state=42)

# Train the classifier
dtree.fit(X_train, y_train)

# Make predictions on the test set
predictions = dtree.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report (precision, recall, F1-score)
print(classification_report(y_test, predictions))

# Create the trained_model directory if it doesn't exist
save_dir = r'..../speech/trained_model'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the trained model
model_name = 'decision_tree_model.joblib'
joblib.dump(dtree, os.path.join(save_dir, model_name))
