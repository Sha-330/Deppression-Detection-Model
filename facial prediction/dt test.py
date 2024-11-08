import joblib
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input

# Load the trained model
model_path = "D:/depression/facial prediction/trained model/decision_tree_model.joblib"
clf = joblib.load(model_path)

# Define emotions and corresponding labels
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Sadness', 5: 'Surprise', 6: 'Neutral'}

# Define depressed and non-depressed emotions with their respective labels
depressed_emotions = {2: 'Fear', 4: 'Sadness'}
non_depressed_emotions = {0: 'Angry', 1: 'Disgust', 3: 'Happiness', 5: 'Surprise', 6: 'Neutral'}

# Function to predict emotion from image
def predict_emotion(img_path):
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        
        feature = model.predict(img_data)
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
        print(f"Error processing image {img_path}: {e}")
        return None, None

# Test the function with an image path
test_img_path = "C:/Users/FMT COMPUTERS/Downloads/dep fresh/images/validation/sad/172.jpg"
predicted_emotion, emotion_type = predict_emotion(test_img_path)
print(f"Predicted emotion for {test_img_path}: {predicted_emotion} ({emotion_type})")
