import cv2
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import os

# Import the architecture from our modular file
# MAKE SURE network.py is in the same folder
from network import ASLNet

# --- 1. HARDWARE DETECTION ---
# We have to verify if the demo is going to run on the CPU or the GPU
def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_best_device()

# --- 2. NORMALIZATION ---
def normalize_landmarks(hand_landmarks):
    points = np.array([[lm.x, lm.y] for lm in hand_landmarks]) # For each pointw in hand_landmarks keep only the (x,y) 
                                                               # and put it in a C like array (faster for computations)
    wrist = points[0].copy() # Set the wrist point as the origin 
    anchored = points - wrist # The points coordinates refers to the distance to the wrist and 
                              # not to the bottom left corner of the camera
    scale = np.linalg.norm(anchored[9]) + 1e-8 # Scale everything by the size of the middle finger  = 
                                               # it won't change the prediction if you are 3 meters away from the camera
    return (anchored / scale).flatten().astype(np.float32)

# --- 3. MAIN APP ---
def run_asl_demo():
    print(f"[+] Starting Universal Engine on: {DEVICE}")
    
    # Initialize the model and load weights
    model = ASLNet(input_size=42, num_classes=26).to(DEVICE)
    try:
        # map_location ensures cross-platform compatibility
        state_dict = torch.load("best_asl_model.pt", map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("[+] Model weights loaded successfully.")
    except Exception as e:
        print(f"[-] Critical Error: Could not load model file. {e}")
        return

    model.eval()

    # MediaPipe Hands Setup (Tasks API)
    model_asset_path = 'hand_landmarker.task'
    if not os.path.exists(model_asset_path):
        print(f"[-] Error: '{model_asset_path}' not found.")
        return

    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5, # if MediaPipe is not sure that he is seeing a hand (<50%) he won't display the points
        min_hand_presence_confidence=0.5, # are you sure the hand is still there ? (<50% ==> don't display)
        running_mode=vision.RunningMode.IMAGE # IMAGE bcz we are running a loop ==> every frame is treated as a single image
    )
    detector = vision.HandLandmarker.create_from_options(options)

    alphabet = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    cap = cv2.VideoCapture(0) # use the default video output of the computer 
    
    # Optimization for laptops                      FOR BETTER PC CONFIGURATIONS ONLY
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)          # 640 --> 1920
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)         # 480 --> 1080
    
    # Define standard MediaPipe hand connections manually (index pairs)
    HAND_CONNECTIONS = [
        # Palm
        (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
        # Fingers
        (1, 2), (2, 3), (3, 4),    # Thumb
        (5, 6), (6, 7), (7, 8),    # Index
        (9, 10), (10, 11), (11, 12), # Middle
        (13, 14), (14, 15), (15, 16), # Ring
        (17, 18), (18, 19), (19, 20)  # Pinky
    ]
    
    prediction_buffer = deque(maxlen=5) # keeps the 5 last frames as reference for letter prediction

    print("[!] Camera starting... Press 'q' to exit.")

    while cap.isOpened(): # As long as the camera works
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1) # Mirrors the video output --> easier to practice ASL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # As OpenCV uses BGR instead of RGB (used by MediaPipe) 
                                                            # we need to translate the data stream because the model is trained based on
                                                            # RGB pictures
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame) # Wrapps the data to make it MediaPipe readable
        
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            coords = normalize_landmarks(landmarks)
            input_tensor = torch.tensor(coords).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)
                
                confidence_score = conf.item() * 100
                prediction_buffer.append(pred_idx.item())

            # Stabilized prediction
            stable_idx = max(set(prediction_buffer), key=prediction_buffer.count)
            letter = alphabet[stable_idx]

            # --- Manual Drawing (Universal) ---
            h, w, _ = frame.shape
            
            # 1. Draw Connections (Lines) first, so points are on top
            # Use white color (BGR: 255, 255, 255) for connections with thickness 2
            for connection in HAND_CONNECTIONS:
                start_point = landmarks[connection[0]]
                end_point = landmarks[connection[1]]
                
                start_coords = (int(start_point.x * w), int(start_point.y * h))
                end_coords = (int(end_point.x * w), int(end_point.y * h))
                
                cv2.line(frame, start_coords, end_coords, (255, 255, 255), 2)
            
            # 2. Draw Landmarks (Points)
            # Use RED color (BGR: 0, 0, 255) with size 5
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                # Optional: Add a white outline for better visibility against red lines
                cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1)

            # Display Letter and Confidence (Green if confident, Orange if not)
            color = (0, 255, 0) if confidence_score > 80 else (0, 165, 255)
            label = f"{letter} ({confidence_score:.1f}%)"
            cv2.putText(frame, label, (30, 70), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)

        cv2.imshow("Universal ASL Recognition - Modular", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    run_asl_demo()