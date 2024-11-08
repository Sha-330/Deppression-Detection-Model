import os
import joblib
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define directories
train_data_dir = "C:/Users/FMT COMPUTERS/Downloads/dep fresh/images/train"
validation_data_dir = "C:/Users/FMT COMPUTERS/Downloads/dep fresh/images/validation"

# Define emotions and corresponding labels
emotion_labels = {'angry': 0, 'sad': 1, 'fear': 2, 'happy': 3, 'surprise': 4, 'neutral': 5, 'disgust': 6}

# Determine number of classes (emotions) in the dataset
num_classes = len(emotion_labels)
print(f"Number of classes (emotions) in the dataset: {num_classes}")

# Function to extract VGGFace features from images
def extract_features(directory):
    print(f"Extracting features from images in {directory}...")
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    features = []
    labels = []
    
    for emotion_folder in os.listdir(directory):
        emotion_path = os.path.join(directory, emotion_folder)
        if not os.path.isdir(emotion_path):
            continue
        emotion_label = emotion_labels.get(emotion_folder)
        if emotion_label is None:
            print(f"Unknown emotion: {emotion_folder}, skipping.")
            continue
        
        for img_file in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_file)
            try:
                img = image.load_img(img_path, target_size=(224, 224))
                img_data = image.img_to_array(img)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                
                feature = model.predict(img_data)
                features.append(feature.flatten())
                labels.append(emotion_label)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
    
    print(f"Feature extraction completed for {len(features)} images.")
    return np.array(features), np.array(labels)

# Extract features and labels from training and validation sets
print("Extracting features from training set...")
train_features, train_labels = extract_features(train_data_dir)
print("Extracting features from validation set...")
val_features, val_labels = extract_features(validation_data_dir)

# Train Decision Tree classifier (C4.5 algorithm)
print("Training C4.5 decision tree classifier...")
clf = DecisionTreeClassifier(criterion='entropy', splitter='best')
clf.fit(train_features, train_labels)
print("Training completed.")

# Predict on validation set and evaluate accuracy
print("Predicting on validation set...")
val_pred = clf.predict(val_features)
accuracy = accuracy_score(val_labels, val_pred)
print(f"Validation Accuracy: {accuracy}")

# Plot confusion matrix
cm = confusion_matrix(val_labels, val_pred)
classes = list(emotion_labels.keys())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

# Plotting confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Save the trained model
model_save_path = "C:/Users/FMT COMPUTERS/Downloads/dep fresh/decision_tree_model.joblib"
joblib.dump(clf, model_save_path)
print(f"Model saved to {model_save_path}")
