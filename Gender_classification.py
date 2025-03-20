import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Parameters
IMG_HEIGHT, IMG_WIDTH = 48, 48  # Image dimensions
N_CHANNELS = 1  # Grayscale images

# Function to load and preprocess images
def load_data(base_dir):
    features = []
    labels = []
    for label, class_dir in enumerate(os.listdir(base_dir)):
        class_path = os.path.join(base_dir, class_dir)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                if file_name.endswith(('.jpg', '.jpeg', '.png')):  # Supported image formats
                    file_path = os.path.join(class_path, file_name)
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
                    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))  # Resize to required dimensions
                    features.append(img)
                    labels.append(label)

    return np.array(features), np.array(labels)

# Define path to your dataset
base_dir = r'C:\Users\Public\Machine learnine project\Emotion_classification\images'

# Check if the directory exists
if not os.path.exists(base_dir):
    raise FileNotFoundError(f"The directory {base_dir} does not exist.")

# Load data
X, y = load_data(base_dir)

# Check if data was loaded correctly
if len(X) == 0 or len(y) == 0:
    raise ValueError("No valid data found. Please check your dataset path or file format.")

# Reshape features to include channel dimension
X = X.reshape(-1, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)

# Normalize pixel values to [0, 1] range
X = X.astype('float32') / 255.0

# Convert labels to one-hot encoding
y = to_categorical(y)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')  # Output layer size matches the number of classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# Save the model
model.save('gender_detection_model.h5')

print("Model training complete and saved as 'image_emotion_detection_model.h5'.")
