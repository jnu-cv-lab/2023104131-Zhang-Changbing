import os
import cv2
import mediapipe as mp
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import config

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_keypoints_from_video(video_path, target_frames=config.TARGET_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            kps = [lm for landmark in results.pose_landmarks.landmark for lm in [landmark.x, landmark.y, landmark.z, landmark.visibility]]
            frames.append(kps)
        else:
            frames.append([0.0] * 132)
    cap.release()
    if not frames:
        return None
    seq = np.array(frames, dtype=np.float32)
    seq = resample_sequence(seq, target_frames)
    seq = normalize_skeleton(seq)
    return seq

def resample_sequence(seq, target_len):
    curr = seq.shape[0]
    if curr == target_len:
        return seq
    idx = np.linspace(0, curr - 1, target_len)
    res = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    for i, t in enumerate(idx):
        low = int(np.floor(t))
        high = min(low + 1, curr - 1)
        if low == high:
            res[i] = seq[low]
        else:
            w = t - low
            res[i] = (1 - w) * seq[low] + w * seq[high]
    return res

def normalize_skeleton(seq):
    left_hip, right_hip = 23, 24
    left_shoulder, right_shoulder = 11, 12
    for t in range(seq.shape[0]):
        coords = seq[t].reshape(-1, 4)[:, :2]
        hip_center = (coords[left_hip] + coords[right_hip]) / 2.0
        shoulder_dist = np.linalg.norm(coords[left_shoulder] - coords[right_shoulder]) or 1.0
        coords = (coords - hip_center) / shoulder_dist
        seq[t, :66] = coords.flatten()
    return seq

def preprocess():
    classes = [d for d in os.listdir(config.RAW_DATA_ROOT) if os.path.isdir(os.path.join(config.RAW_DATA_ROOT, d))]
    label_map = {cls: i for i, cls in enumerate(classes)}
    os.makedirs(config.PREPROCESSED_DATA_DIR, exist_ok=True)
    with open(os.path.join(config.PREPROCESSED_DATA_DIR, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=4)
    
    all_X, all_y = [], []
    for cls, cid in label_map.items():
        path = os.path.join(config.RAW_DATA_ROOT, cls)
        videos = [v for v in os.listdir(path) if v.endswith(('.mp4','.avi','.mov','.mkv'))]
        for vid in tqdm(videos, desc=f"处理 {cls}"):
            seq = extract_keypoints_from_video(os.path.join(path, vid))
            if seq is not None:
                all_X.append(seq)
                all_y.append(cid)
    X = np.array(all_X)
    y = np.array(all_y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    np.save(os.path.join(config.PREPROCESSED_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(config.PREPROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(config.PREPROCESSED_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(config.PREPROCESSED_DATA_DIR, 'y_test.npy'), y_test)
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

if __name__ == '__main__':
    preprocess()