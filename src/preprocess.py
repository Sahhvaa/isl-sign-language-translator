import os
import numpy as np
from extract_keypoints import build_landmarkers, extract_keypoints

MAX_FRAMES = 30

def pad_or_trim(seq, max_len=MAX_FRAMES):
    if len(seq) >= max_len:
        return seq[:max_len]
    pad = np.zeros((max_len - len(seq), seq.shape[1]))
    return np.vstack([seq, pad])

def build_dataset():
    labels = sorted(os.listdir("data/raw_videos"))
    X, y = [], []

    for label in labels:
        folder = f"data/raw_videos/{label}"
        for video_file in os.listdir(folder):
            pose_landmarker, hand_landmarker = build_landmarkers()  # fresh per video
            kp = extract_keypoints(f"{folder}/{video_file}", pose_landmarker, hand_landmarker)
            pose_landmarker.close()
            hand_landmarker.close()

            X.append(pad_or_trim(kp))
            y.append(labels.index(label))
            print(f"Processed {label}/{video_file} -> shape {kp.shape}")

    X = np.array(X)
    y = np.array(y)
    np.save("data/X.npy", X)
    np.save("data/y.npy", y)
    print("Final dataset shape:", X.shape, y.shape)
    return labels

if __name__ == "__main__":
    labels = build_dataset()
    print("Labels:", labels)