"""
Real-time ASL letter recognition from the webcam.

Usage: python demo.py [--model best_asl_model.pt] [--camera 0]     (press 'q' to quit)
"""
import argparse
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from landmarks import ALPHABET, DEFAULT_MODEL_PATH, HAND_CONNECTIONS, HAND_MODEL_PATH, normalize_landmarks
from network import ASLNet, load_model

BUFFER_SIZE = 5                 # frames used to stabilize the prediction
CONFIDENT_THRESHOLD = 80.0      # % above which the label is drawn in green
FRAME_WIDTH, FRAME_HEIGHT = 640, 480   # raise to 1280x720 / 1920x1080 on powerful PCs
WINDOW_NAME = "Universal ASL Recognition - Modular"


def get_best_device() -> torch.device:
    """ CUDA (NVIDIA) > MPS (Apple Silicon) > CPU """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_detector() -> vision.HandLandmarker | None:
    """ VIDEO mode tracks the hand between frames: faster and steadier than IMAGE mode. """
    if not HAND_MODEL_PATH.exists():
        print(f"[-] Error: '{HAND_MODEL_PATH}' not found.")
        return None
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        num_hands=1,
        min_hand_detection_confidence=0.5,   # below 50% sure it is a hand -> ignored
        min_hand_presence_confidence=0.5,    # below 50% sure the hand is still there -> re-detect
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(options)


@torch.no_grad()
def predict(model: torch.nn.Module, landmarks, device: torch.device) -> np.ndarray:
    """ Returns the 26 class probabilities for one hand. """
    x = torch.from_numpy(normalize_landmarks(landmarks)).unsqueeze(0).to(device)
    return torch.softmax(model(x), dim=1)[0].cpu().numpy()


def stabilize(buffer: deque) -> tuple[str, float]:
    """ Majority vote over recent frames; confidence = mean probability of the winning letter. """
    winner = Counter(int(np.argmax(p)) for p in buffer).most_common(1)[0][0]
    confidence = float(np.mean([p[winner] for p in buffer])) * 100
    return ALPHABET[winner], confidence


def draw_hand(frame: np.ndarray, landmarks) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:                       # lines first, points on top
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for p in pts:
        cv2.circle(frame, p, 5, (0, 0, 255), -1)
        cv2.circle(frame, p, 6, (255, 255, 255), 1)


def draw_label(frame: np.ndarray, letter: str, confidence: float) -> None:
    color = (0, 255, 0) if confidence > CONFIDENT_THRESHOLD else (0, 165, 255)
    cv2.putText(frame, f"{letter} ({confidence:.1f}%)", (30, 70), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)


def run_loop(cap: cv2.VideoCapture, detector: vision.HandLandmarker,
             model: torch.nn.Module, device: torch.device) -> None:
    buffer = deque(maxlen=BUFFER_SIZE)
    last_ts = -1
    while True:
        success, frame = cap.read()
        if not success:
            print("[-] Camera stopped sending frames.")
            break

        frame = cv2.flip(frame, 1)                              # mirror: easier to practice
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)            # OpenCV is BGR, MediaPipe wants RGB
        ts = max(int(time.monotonic() * 1000), last_ts + 1)     # VIDEO mode needs increasing timestamps
        last_ts = ts
        result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            buffer.append(predict(model, landmarks, device))
            draw_hand(frame, landmarks)
            draw_label(frame, *stabilize(buffer))
        else:
            buffer.clear()          # don't mix old predictions with a new hand sign

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def run_asl_demo(model_path: str, camera: int) -> int:
    device = get_best_device()
    print(f"[+] Starting Universal Engine on: {device}")

    model = ASLNet().to(device)
    if not load_model(model, model_path, device):
        return 1
    model.eval()

    detector = create_detector()
    if detector is None:
        return 1

    cap = cv2.VideoCapture(camera)
    try:
        if not cap.isOpened():
            print(f"[-] Error: could not open camera {camera}.")
            return 1
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print("[!] Camera starting... Press 'q' to exit.")
        run_loop(cap, detector, model, device)
        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default=str(DEFAULT_MODEL_PATH))
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args(argv)
    return run_asl_demo(args.model, args.camera)


if __name__ == "__main__":
    raise SystemExit(main())
