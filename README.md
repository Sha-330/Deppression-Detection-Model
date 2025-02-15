# Emotion Detection System  

This project was developed by our 3-member team as computer science( AI & DS ) students. Our initial goal was to build an ML model capable of predicting depression from audio, text, and video and hosting it on a user-friendly website. However, after development, we could implement only an emotion detection system.  

## Features  

### 1. Emotion Detection from Facial Expressions  
We implemented emotion detection in two ways:  
- **Captured Images and Videos**: The system analyzes images and pre-recorded videos to detect emotions.  
- **Live Image and Video Processing**: The model processes real-time video feeds to detect emotions dynamically.  

For facial emotion detection, we used **Haarcascade**, which analyzes emotions in each frame of the input.  

### 2. Emotion Detection from Audio  
- We used a **Decision Tree model** to analyze variations in voice modulation.  
- The system compares uploaded audio with a baseline to detect emotional shifts.  

### 3. Emotion Detection from Text  
- The model processes user-provided text, analyzing it **word by word**.  
- It checks for words categorized under "sad" or "depression-related" (e.g., *loneliness, sad*).  

## Website  
To provide a user-friendly interface, we created a simple website using **HTML and CSS** with a minimalistic theme.  

## Future Improvements  
Since this was our first project, we acknowledge there may be some mistakes. However, we are committed to improving and refining our approach in future iterations.  
