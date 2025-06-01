# Depression Detection System

## Overview
The **Depression Detection System** is a machine learning-based application that detects emotions from **facial expressions, audio recordings, and text input**. This system integrates different AI models to analyze human emotions, aiding in mental health assessment.

## Features
- **Facial Emotion Detection**: Uses CNN and VGG16 to analyze facial expressions from images.
- **Audio Emotion Detection**: Extracts MFCC features from speech and classifies emotions using a Decision Tree model.
- **Text Emotion Detection**: Processes text input and predicts emotions using a Decision Tree classifier with Bag-of-Words representation.
- **Model Saving & Evaluation**: All trained models are stored for future inference, and evaluation metrics like confusion matrix and accuracy are provided.

## Installation
### Prerequisites
Ensure you have the following libraries installed:
```bash
pip install tensorflow keras numpy matplotlib scikit-learn librosa pandas joblib
```

## Facial Emotion Detection
### Model Training
1. Dataset is stored in directories for training and validation.
2. CNN-based model is trained using `tensorflow.keras`.
3. The trained model is saved as `cnn_emotion_model.h5`.

### Code Snippet
```python
cnn_model.save("cnn_emotion_model.h5")
print("Model saved successfully!")
```

## Audio Emotion Detection
### Model Training
1. Extracts **MFCC** features from audio.
2. Uses a **Decision Tree Classifier** for classification.
3. Saves the trained model as `decision_tree_model.joblib`.

### Code Snippet
```python
joblib.dump(dtree, os.path.join(save_dir, 'decision_tree_model.joblib'))
print("Audio model saved successfully!")
```

## Text Emotion Detection
### Model Training
1. Loads multiple text emotion datasets.
2. Cleans and vectorizes text using `CountVectorizer`.
3. Trains a **Decision Tree Classifier** and saves the model.

### Code Snippet
```python
joblib.dump(clf, model_save_path)
joblib.dump(vectorizer, vectorizer_save_path)
print("Text emotion model saved successfully!")
```

## Usage
Once trained, models can be loaded for inference:
```python
from tensorflow.keras.models import load_model
model = load_model("cnn_emotion_model.h5")
```
For audio and text detection:
```python
clf = joblib.load("emotion_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
```

## Results & Evaluation
- **Facial Emotion Detection**: Displays confusion matrix.
- **Audio Emotion Detection**: Prints accuracy score.
- **Text Emotion Detection**: Outputs predicted emotion for a sample text.


## Contributors
- **Mazin Muneer (mazi)** - [GitHub](https://github.com/maaazzinn) | [LinkedIn](https://www.linkedin.com/in/mazin-muneer?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BdvLryXBiQjypj5RZtQSCow%3D%3D)
- **Ajmal Shan. P (Sha)** - [GitHub](https://github.com/Sha-330) | [LinkedIn](https://www.linkedin.com/in/ajmal-shan-p-591258244)
- **Adam Nahan(Adam P)** - [GitHub]((https://github.com/adamnahan)) | [LinkedIn](https://www.linkedin.com/in/adam-nahan-34a95524a?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BCLRzX0qSRBC%2FrcGGVwgkQw%3D%3D)

