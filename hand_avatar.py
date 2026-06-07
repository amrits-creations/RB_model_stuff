import argparse
import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from models import MODELS

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

NUM_LANDMARKS = 21
INPUT_DIM = NUM_LANDMARKS * 2

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def draw_hand(frame, coords, color, offset_x=0):
    h, w, _ = frame.shape
    pts = []
    for i in range(NUM_LANDMARKS):
        x = int(coords[i*2] * w) + offset_x
        y = int(coords[i*2 + 1] * h)
        pts.append((x, y))
        cv2.circle(frame, (x, y), 4, color, -1)

    for (idx1, idx2) in HAND_CONNECTIONS:
        if pts[idx1] != (offset_x, 0) and pts[idx2] != (offset_x, 0):
            cv2.line(frame, pts[idx1], pts[idx2], color, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), default="single")
    args = parser.parse_args()

    download_model()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    detector = vision.HandLandmarker.create_from_options(options)

    hidden_dim = 16
    pcn = MODELS[args.model](INPUT_DIM, hidden_dim) # initialize the model with the input and hidden dims.
    avatar_state = np.ones((INPUT_DIM, 1)) * 0.5 # this is the initial state of the avatar - we start it at the center pose (all 0.5s) and let it learn from there. The model will update this state over time to try to match the user's hand pose.
    action_lr = 0.25

    TRAINING_FRAMES = 150
    current_frame = 0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Active Inference Engine Online. Press 'q' to halt.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV rerads images in BGR but MediaPipe expects RGB

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)

            display_frame = np.zeros_like(frame)
            h, w, _ = frame.shape
            half_w = w // 2

            if detection_result.hand_landmarks:
                hand_landmarks = detection_result.hand_landmarks[0]

                user_coords = []
                for lm in hand_landmarks:
                    user_coords.extend([lm.x, lm.y])

                Y_user = np.array(user_coords).reshape(INPUT_DIM, 1)
                r_td = np.dot(pcn.U.T, Y_user) # I don't like this line. I won't explain why. I just don't.

                if current_frame < TRAINING_FRAMES:
                    # PHASE 1: PRE-TRAINING (Observation Only)

                    for _ in range(3):
                        pcn.update_state(Y_user, r_td, dt=0.1)
                    pcn.update_weights(Y_user, dt=0.1)

                    cv2.putText(display_frame, f"TRAINING: WAVE HAND ({current_frame}/{TRAINING_FRAMES})", (half_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    current_frame += 1
                else:
                    # PHASE 2: ACTIVE INFERENCE

                    for _ in range(3):
                        pcn.update_state(avatar_state, r_td, dt=0.1) # Removed I_avatar because it was put in by the clanker and very obviously redundant.
                    pcn.update_weights(avatar_state, dt=0.1)

                    prediction = pcn.get_prediction()
                    avatar_state += action_lr * (prediction - avatar_state)

                    cv2.putText(display_frame, "PCN AVATAR (ACTIVE)", (half_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                # --- UI Rendering ---
                scaled_user = Y_user.copy()
                for i in range(NUM_LANDMARKS): scaled_user[i*2] *= 0.5
                draw_hand(display_frame, scaled_user.flatten(), (0, 255, 0))

                scaled_avatar = avatar_state.copy()
                for i in range(NUM_LANDMARKS): scaled_avatar[i*2] *= 0.5
                draw_hand(display_frame, scaled_avatar.flatten(), (0, 165, 255), offset_x=half_w)

            cv2.putText(display_frame, "USER TARGET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.line(display_frame, (half_w, 0), (half_w, h), (50, 50, 50), 2)

            cv2.imshow("PCN Hand Avatar", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Inference Halted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
