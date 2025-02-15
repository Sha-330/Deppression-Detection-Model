import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# File paths for datasets
emotion_dataset_path = r'..../text prediction/dataset/emotion_dataset.csv'
text_emotion_path = r'..../text prediction/dataset/text_emotion.csv'
tweet_emotions_path = r'..../text prediction/dataset/tweet_emotions.csv'

try:
    # Load datasets
    emotion_dataset = pd.read_csv(emotion_dataset_path)
    text_emotion = pd.read_csv(text_emotion_path)
    tweet_emotions = pd.read_csv(tweet_emotions_path)

    # Select relevant columns and rename to match
    emotion_dataset = emotion_dataset[['Emotion', 'Clean_Text']].rename(columns={'Clean_Text': 'text', 'Emotion': 'emotion'})
    text_emotion = text_emotion[['sentiment', 'content']].rename(columns={'content': 'text', 'sentiment': 'emotion'})
    tweet_emotions = tweet_emotions[['sentiment', 'content']].rename(columns={'content': 'text', 'sentiment': 'emotion'})

    # Concatenate datasets
    combined_df = pd.concat([emotion_dataset, text_emotion, tweet_emotions], ignore_index=True)

    # Preprocess text (example: remove punctuation)
    combined_df['text'] = combined_df['text'].str.replace('[^\w\s]', '')

    # Feature extraction using Bag-of-Words
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(combined_df['text'].fillna('').values.astype('str'))
    y = combined_df['emotion'].fillna('').values.astype('str')  # Adjust according to your target column

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Decision Tree classifier (C4.5 algorithm)
    clf = DecisionTreeClassifier()

    # Train the classifier
    clf.fit(X_train, y_train)

    # Save the trained model and vectorizer
    model_save_path = r'...../text prediction/trained model/emotion_classifier_model.pkl'
    vectorizer_save_path = r'...../text prediction/trained model/vectorizer.pkl'
    
    joblib.dump(clf, model_save_path)
    joblib.dump(vectorizer, vectorizer_save_path)

    # Predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Example prediction
    new_text = ["I feel excited"]
    new_text_transformed = vectorizer.transform(new_text)
    predicted_emotion = clf.predict(new_text_transformed)
    print("Predicted emotion:", predicted_emotion)

except Exception as e:
    print("Error:", e)
