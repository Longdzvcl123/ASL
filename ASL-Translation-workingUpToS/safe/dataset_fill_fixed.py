"""
Records hand landmarks from the webcam into a CSV (one row per frame).

Usage: python dataset_fill_fixed.py [--csv asl_dataset_varied.csv] [--frames 1300] [--camera 0]
Then:  python pickle_data.py --csv asl_dataset_varied.csv
"""
import argparse
import csv
import os
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from landmarks import BASE_DIR, HAND_MODEL_PATH, NUM_LANDMARKS, label_to_index

FRAME_WIDTH, FRAME_HEIGHT = 640, 480     # keep identical to demo.py (x/y depend on aspect ratio)
COUNTDOWN_SEC = 2
FRAMES_PER_PHASE = 200
ESC_KEY = 27
PHASES = [
    "Hold steady - neutral position",
    "Move slowly left/right",
    "Move slowly up/down",
    "Tilt hand forward/back",
    "Rotate hand slightly",
    "Vary distance from camera",
]


def get_variation_phase(frame_count: int) -> str:
    """ Variation instruction based on progress through the recording. """
    return PHASES[(frame_count // FRAMES_PER_PHASE) % len(PHASES)]


def create_detector() -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(options)


def ensure_header(csv_file: str | os.PathLike) -> None:
    if os.path.exists(csv_file):
        return
    headers = ['label']
    for i in range(NUM_LANDMARKS):
        headers.extend([f'x{i}', f'y{i}', f'z{i}'])
    with open(csv_file, mode='w', newline='') as f:
        csv.writer(f).writerow(headers)


def ask_label() -> str | None:
    """ Returns a valid letter A-Z, or None to quit. """
    while True:
        label = input("\nEnter the ASL letter you want to record (or 'quit' to quit): ").strip().upper()
        if label == 'QUIT':
            return None
        if label_to_index(label) is not None:
            return label
        print("[!] Please enter a single letter A-Z.")


def record_letter(cap: cv2.VideoCapture, detector: vision.HandLandmarker, label: str,
                  csv_file: str, frames_per_letter: int) -> int:
    frames_collected, last_ts = 0, -1
    with open(csv_file, mode='a', newline='') as f:       # opened once, not every frame
        writer = csv.writer(f)
        while frames_collected < frames_per_letter:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = max(int(time.monotonic() * 1000), last_ts + 1)
            last_ts = ts
            result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            if result.hand_landmarks:
                row = [label]
                for lm in result.hand_landmarks[0]:
                    row.extend([lm.x, lm.y, lm.z])
                    cv2.circle(frame, (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])), 5, (0, 255, 0), -1)
                writer.writerow(row)
                frames_collected += 1

            progress = frames_collected / frames_per_letter * 100
            cv2.putText(frame, f"Collecting '{label}': {frames_collected}/{frames_per_letter} ({progress:.0f}%)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Tip: {get_variation_phase(frames_collected)}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow('ASL Data Collection', frame)

            if cv2.waitKey(1) & 0xFF == ESC_KEY:
                print("Recording interrupted.")
                break
    return frames_collected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', default=str(BASE_DIR / 'asl_dataset_varied.csv'))
    parser.add_argument('--frames', type=int, default=1300)
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args(argv)
    if args.frames < 1:
        parser.error('--frames must be >= 1')

    if not HAND_MODEL_PATH.exists():
        sys.exit(f"ERROR: '{HAND_MODEL_PATH}' not found! Please download it into this folder before running.")

    ensure_header(args.csv)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("ERROR: Could not open webcam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    detector = create_detector()

    print("=== ASL Data Collection ===\nPress ESC to stop recording early for current letter")
    try:
        while (label := ask_label()) is not None:
            print(f"Recording starts in {COUNTDOWN_SEC} seconds for '{label}'")
            time.sleep(COUNTDOWN_SEC)
            count = record_letter(cap, detector, label, args.csv, args.frames)
            print(f"Successfully recorded {count} frames for '{label}'.")
            cv2.destroyAllWindows()
            cv2.waitKey(1)
    except KeyboardInterrupt:
        print("\nData collection interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print(f"\nData collection complete. Data saved to: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
