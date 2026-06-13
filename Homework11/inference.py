import torch
import numpy as np
import json
import sys
import config
from model import SkeletonTransformer
from preprocess import extract_keypoints_from_video

def inference(video_path):
    seq = extract_keypoints_from_video(video_path, config.TARGET_FRAMES)
    if seq is None:
        print("未检测到人体骨架")
        return
    with open(f"{config.PREPROCESSED_DATA_DIR}/label_map.json", 'r') as f:
        label_map = json.load(f)
    inv_map = {v: k for k, v in label_map.items()}
    
    model = SkeletonTransformer().to(config.DEVICE)
    model.load_state_dict(torch.load(f"{config.CHECKPOINT_DIR}/best_model.pth", map_location=config.DEVICE))
    model.eval()
    
    with torch.no_grad():
        logits = model(torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(config.DEVICE))
        prob = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(prob).item()
        conf = prob[pred].item()
    print(f"Predicted: {inv_map[pred]}, Confidence: {conf:.4f}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("/home/ayn/Homework12/badminton_stroke_video")
        sys.exit(1)
    inference(sys.argv[1])