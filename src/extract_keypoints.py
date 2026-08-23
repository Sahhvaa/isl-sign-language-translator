import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions,
    RunningMode,
)

def build_landmarkers():
    pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/pose_landmarker.task"),
        running_mode=RunningMode.VIDEO, num_poses=1))
    hands = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
        running_mode=RunningMode.VIDEO, num_hands=2))
    return pose, hands

def flatten_landmarks(landmark_list, expected_points):
    if not landmark_list:
        return np.zeros(expected_points * 3)
    pts = landmark_list[0]
    return np.array([[p.x, p.y, p.z] for p in pts]).flatten()

def extract_keypoints(video_path, pose_landmarker, hand_landmarker):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sequence, frame_idx = [], 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((frame_idx / fps) * 1000)

        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        pose_vec = flatten_landmarks(pose_result.pose_landmarks, 33)   # 99 numbers
        hl = hand_result.hand_landmarks
        hand1 = flatten_landmarks(hl[0:1], 21) if len(hl) > 0 else np.zeros(63)
        hand2 = flatten_landmarks(hl[1:2], 21) if len(hl) > 1 else np.zeros(63)

        sequence.append(np.concatenate([pose_vec, hand1, hand2]))   # 99+63+63 = 225
        frame_idx += 1

    cap.release()
    return np.array(sequence)   # shape: (num_frames, 225)


if __name__ == "__main__":
    pose_landmarker, hand_landmarker = build_landmarkers()
    kp = extract_keypoints("data/your_sign_video.mp4", pose_landmarker, hand_landmarker)
    print("Extracted shape:", kp.shape)
