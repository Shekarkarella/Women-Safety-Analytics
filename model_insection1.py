import os
import numpy as np
import cv2
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained models
audio_model = load_model('audio_classification_model.h5')  # Model for audio classification
gender_model = load_model('gender_classification_model.h5')  # Model for gender classification from video
emotion_image_model = load_model('image_emotion_detection_model.h5')  # Model for emotion detection from images

# Parameters
IMG_HEIGHT, IMG_WIDTH = 48, 48  # Image dimensions for image-based models
N_MFCC = 13  # Number of MFCC features for audio

# Function to preprocess audio input for emotion detection
def preprocess_audio(audio_path, duration=2, sample_rate=22050):
    signal, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
    
    # Ensure fixed length by padding or truncating
    if len(signal) < sample_rate * duration:
        signal = np.pad(signal, (0, sample_rate * duration - len(signal)), mode='constant')
    else:
        signal = signal[:sample_rate * duration]
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=N_MFCC)
    mfcc = np.mean(mfcc.T, axis=0)  # Take mean of MFCCs across time
    return np.expand_dims(mfcc, axis=0)  # Return MFCC feature ready for model input

# Function to preprocess video frames for gender detection
def preprocess_video_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    resized_frame = cv2.resize(gray_frame, (IMG_HEIGHT, IMG_WIDTH))  # Resize frame
    return np.expand_dims(np.expand_dims(resized_frame, axis=-1), axis=0)  # Reshape to match model input

# Function to preprocess image input for emotion detection
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
    resized_img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))  # Resize image
    return np.expand_dims(np.expand_dims(resized_img, axis=-1), axis=0)  # Reshape to match model input

# Function to classify gender from video frame
def classify_gender_from_video(video_frame):
    preprocessed_frame = preprocess_video_frame(video_frame)
    gender_prediction = gender_model.predict(preprocessed_frame)
    gender_label = np.argmax(gender_prediction, axis=1)[0]  # 0 for female, 1 for male
    return 'Male' if gender_label == 1 else 'Female'

# Function to classify emotion from audio
def classify_emotion_from_audio(audio_path):
    preprocessed_audio = preprocess_audio(audio_path)
    audio_prediction = audio_model.predict(preprocessed_audio)
    emotion_label = np.argmax(audio_prediction, axis=1)[0]
    return f'Audio Emotion Class: {emotion_label}'

# Function to classify emotion from image
def classify_emotion_from_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    emotion_prediction = emotion_image_model.predict(preprocessed_image)
    emotion_label = np.argmax(emotion_prediction, axis=1)[0]
    return f'Image Emotion Class: {emotion_label}'

# Example usage:
# Assuming 'video_frame' is a single frame from video, 'audio_path' is the path to an audio file,
# and 'image_path' is the path to an image file for emotion detection

video_frame = cv2.imread('sample_video_frame.jpg')  # Replace with actual video frame
audio_path = 'sample_audio.wav'  # Replace with actual audio file path
image_path = 'sample_image.jpg'  # Replace with actual image file path

# Gender classification from video
gender_result = classify_gender_from_video(video_frame)
print(gender_result)

# Emotion classification from audio
audio_emotion_result = classify_emotion_from_audio(audio_path)
print(audio_emotion_result)

# Emotion classification from image
image_emotion_result = classify_emotion_from_image(image_path)
print(image_emotion_result)
