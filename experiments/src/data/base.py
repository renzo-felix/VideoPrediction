import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from .video_loaders import load_video, normalize_video

class VideoDataset(Dataset):
    def __init__(self, metadata_csv, videos_base_dir=None, img_size=224, num_frames=16):
        self.df = pd.read_csv(metadata_csv)
        self.videos_base_dir = Path(videos_base_dir) if videos_base_dir else None
        self.img_size = img_size
        self.num_frames = num_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row['video_id']

        if 'video_path' in row:
            video_path = Path(row['video_path'])
            if not video_path.is_absolute() and self.videos_base_dir:
                video_path = self.videos_base_dir / video_path

            if not video_path.exists() and 'category' in row and self.videos_base_dir:
                number = video_id.rsplit('_', 1)[-1]
                category = row['category']
                variation = row.get(category, '')

                if variation:
                    video_id = f"{category}_{variation}_{number}"
                    video_path = self.videos_base_dir / 'videos' / f"test_{category}" / variation / f"{video_id}.mp4"
        else:
            if not self.videos_base_dir:
                raise ValueError("Either video_path column or videos_base_dir must be provided")
            video_path = self.videos_base_dir / 'videos' / f"{video_id}.mp4"

        frames = load_video(video_path, self.img_size, self.num_frames)
        frames = normalize_video(frames)

        return {
            'video_id': video_id,
            'frames': frames,
            'actual_speed': row['actual_speed'],
            'target_speed': row['target_speed']
        }
