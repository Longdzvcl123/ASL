import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision   
import csv
import os
import time
model_path = 'hand_landmarker.task'

if not os.path.exists(model_path):
    print(f"ERROR: '{model_path}' not found!")
    print("Please download it into this folder before running.")
    exit()

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    running_mode=vision.RunningMode.IMAGE 
)

detector = vision.HandLandmarker.create_from_options(options)

TRACKERS = 21 
csv_file = 'asl_dataseeet.csv'
frames_per_letter = 800 
	
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        headers = ['label']
        for i in range(TRACKERS): 
            headers.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(headers)

cap = cv2.VideoCapture(0)

while True:
    label = input("\nEnter the ASL letter you want to record (or 'quit' to quit): ").upper()
    
    if label.lower() == 'quit':
        print("Exiting data collection.")
        break

    print(f"recording starts in 3 seconds")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print(f"Recording {frames_per_letter} frames")

    frames_collected = 0

    while frames_collected < frames_per_letter:
        ret, frame = cap.read()
        if not ret: 
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1) 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks: 
            hand_landmarks = detection_result.hand_landmarks[0]

            row = [label]
            for landmark in hand_landmarks:
                row.extend([landmark.x, landmark.y, landmark.z])

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            frames_collected += 1

        cv2.putText(frame, f"Collecting '{label}': {frames_collected}/{frames_per_letter}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('ASL Data Collection', frame)

        if cv2.waitKey(1) & 0xFF == 27: 
            print("Recording interrupted.")
            break

    print(f"Successfully recorded {frames_collected} frames for '{label}'.")
    cv2.destroyAllWindows()
    cv2.waitKey(1) 
    
cap.release()
cv2.destroyAllWindows()
detector.close()