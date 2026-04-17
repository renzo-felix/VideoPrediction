import cv2
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms

def load_video(video_path, img_size=224, num_frames=16):
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames: {video_path}")

    if len(frames) > num_frames:
        indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]
    elif len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))

    return np.array(frames)

def normalize_video(frames):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return torch.stack([transform(f) for f in frames])
