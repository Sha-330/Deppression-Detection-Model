import joblib

# Load the trained model and vectorizer
model_load_path = r'D:/Telegram Desktop/depression/text prediction/trained model/emotion_classifier_model.pkl'
vectorizer_load_path = r'D:/Telegram Desktop/depression/text prediction/trained model/vectorizer.pkl'

# Define depressed and non-depressed emotions
depressed_emotions = ['sadness', 'emptiness', 'worry', 'shame']
non_depressed_emotions = ['happiness', 'joy', 'relief', 'surprise', 'enthusiasm', 'love', 'neutral', 'anger', 'boredom', 'disgust', 'hate', 'fear']

try:
    # Load the model and vectorizer
    clf = joblib.load(model_load_path)
    vectorizer = joblib.load(vectorizer_load_path)

    # Example prediction
    new_text = ["I feel excited"]
    new_text_transformed = vectorizer.transform(new_text)
    predicted_emotion = clf.predict(new_text_transformed)[0]
    print("Predicted emotion:", predicted_emotion)

    # Determine if predicted emotion is depressed or not
    if predicted_emotion.lower() in depressed_emotions:
        print("Predicted emotion is depressed.")
    elif predicted_emotion.lower() in non_depressed_emotions:
        print("Predicted emotion is not depressed.")
    else:
        print("Predicted emotion category is unknown.")

except Exception as e:
    print("Error:", e)
