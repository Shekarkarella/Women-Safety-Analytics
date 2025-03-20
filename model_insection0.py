import cv2
import numpy as np
import random
import time
import pyaudio
import struct

# Emotion and gender labels
emotion_labels = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
gender_labels = ['Male', 'Female']  # Gender labels

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    raise IOError('Failed to load face cascade classifier')

# Initialize variables
last_emotion_update_time = time.time()
current_emotions = []
previous_faces = []  # Initialize previous_faces
motion_threshold = 50  # Threshold for detecting heavy movement
age_last_update_time = time.time()
gender_last_update_time = time.time()  # To track gender update delay
assigned_ages = []  # Store the current age assignments
assigned_genders = []  # Store the current gender assignments
emotion_update_interval = 4  # 4 seconds delay for updating emotions
age_change_interval = 4  # 4 seconds delay for updating age and gender
activity_last_update_time = time.time()  # Time of last activity status update
activity_update_interval = 2.5  # 2.5 seconds delay before updating activity status

# Audio detection parameters
THRESHOLD = 500
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# High Alert variables
high_alert_active = False
high_alert_start_time = 0
high_alert_duration = 2  # 2 seconds for high alert

def update_emotions(num_faces):
    global current_emotions
    current_emotions = [random.choice(emotion_labels) for _ in range(num_faces)]

def assign_gender():
    return random.choice(gender_labels)

def assign_age():
    age_ranges = ['20-25', '29-35']
    return random.choice(age_ranges)

def detect_noise():
    try:
        data = stream.read(CHUNK)
        audio_data = np.array(struct.unpack(str(CHUNK) + 'h', data), dtype='int16')
        audio_level = np.abs(audio_data).mean()
        return audio_level > THRESHOLD
    except Exception as e:
        print(f"Error in detect_noise: {e}")
        return False

def detect_activity_level(faces):
    global previous_faces
    if len(faces) < 1:
        return 'Normal environment'

    significant_movement = False
    high_movement = False

    # Convert face coordinates to a list of positions
    current_face_positions = [face[:2] for face in faces]

    # Debug statement to check the number of faces
    print(f"Current number of faces: {len(faces)}")
    print(f"Previous number of faces: {len(previous_faces)}")

    # Ensure previous_faces has the same number of faces for comparison
    if len(previous_faces) == len(faces):
        for i, (x, y, w, h) in enumerate(faces):
            prev_x, prev_y, _, _ = previous_faces[i]
            # Calculate movement distance
            movement = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
            # Debug statement to check movement
            print(f"Face {i} movement: {movement}")

            if movement > motion_threshold:
                significant_movement = True
                if movement > 2 * motion_threshold:
                    high_movement = True
    else:
        significant_movement = True  # New faces detected

    # Update previous_faces with current face positions
    previous_faces[:] = faces

    if high_movement:
        return 'Abnormal activity'
    elif significant_movement:
        return 'Miscellaneous activity'
    else:
        return 'Normal environment'

# Initialize the previous activity level
previous_activity_level = 'Normal environment'

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    current_time = time.time()

    # Update emotions every 4 seconds
    if current_time - last_emotion_update_time >= emotion_update_interval:
        update_emotions(len(faces))
        last_emotion_update_time = current_time

    # Update ages and genders every 4 seconds
    if current_time - age_last_update_time >= age_change_interval:
        assigned_ages = [assign_age() for _ in range(len(faces))]
        age_last_update_time = current_time

    if current_time - gender_last_update_time >= age_change_interval:
        assigned_genders = [assign_gender() for _ in range(len(faces))]
        gender_last_update_time = current_time

    male_count = 0
    female_count = 0

    for i, (x, y, w, h) in enumerate(faces):
        gender = assigned_genders[i] if i < len(assigned_genders) else assign_gender()
        age = assigned_ages[i] if i < len(assigned_ages) else assign_age()

        # Count male and female
        if gender == 'Male':
            male_count += 1
        else:
            female_count += 1

        emotion = current_emotions[i] if i < len(current_emotions) else random.choice(emotion_labels)

        # Draw the rectangle and text for each face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'{gender}, Age: {age}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the number of faces detected
    num_faces = len(faces)
    text_people = f'People detected: {num_faces}'
    text_x = 10
    text_y = frame.shape[0] - 80  # Adjust y position
    cv2.putText(frame, text_people, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the gender ratio at the top left corner
    if female_count > 0:
        gender_ratio = f'Gender Ratio: {male_count}:{female_count} (Male:Female)'
    else:
        gender_ratio = f'Gender Ratio: {male_count}:0 (Male:Female)'
    
    cv2.putText(frame, gender_ratio, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Determine the current activity level
    current_activity_level = detect_activity_level(faces)

    # Check if enough time has passed since the last activity status update
    if current_time - activity_last_update_time >= activity_update_interval:
        if current_activity_level != previous_activity_level:
            previous_activity_level = current_activity_level
            activity_last_update_time = current_time

    text = previous_activity_level
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_x_activity = frame.shape[1] - text_size[0] - 10
    text_y_activity = frame.shape[0] - 10

    # Set color based on activity level
    if previous_activity_level == 'Abnormal activity':
        color = (0, 0, 255)  # Red
    elif previous_activity_level == 'Miscellaneous activity':
        color = (0, 255, 255)  # Yellow
        high_alert_active = True  # Trigger high alert when miscellaneous activity is detected
        high_alert_start_time = current_time  # Record the start time of high alert
    else:
        color = (0, 255, 0)  # Green

    cv2.putText(frame, text, (text_x_activity, text_y_activity), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Check if high alert is active and display it for 2 seconds
    if high_alert_active and current_time - high_alert_start_time <= high_alert_duration:
        cv2.putText(frame, '!!!HIGH ALERT!!!', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    elif current_time - high_alert_start_time > high_alert_duration:
        high_alert_active = False  # Deactivate high alert after 2 seconds

    cv2.imshow('Activity Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
p.terminate()