from torch.utils.data import DataLoader, IterableDataset
import torch
import torch.nn.functional as F

import os
import random

class CoDDataset(IterableDataset):
    def __init__(self, root = "../cod_data/BlackOpsColdWar"):
        super().__init__()

        self.paths = []
        for root_dir in os.listdir(root):
            splits_dir = os.path.join(root, root_dir, "splits")
            if not os.path.isdir(splits_dir):
                continue
                
            # Get all files in splits dir
            files = os.listdir(splits_dir)
            # Filter to just the RGB files
            rgb_files = [f for f in files if f.endswith("_rgb.pt")]
            
            for rgb_file in rgb_files:
                rgb_path = os.path.join(splits_dir, rgb_file)
                self.paths.append(rgb_path)
    
    def get_item(self):
        # Pick random video path
        vid_path = random.choice(self.paths)
        # Load video tensor with memory mapping
        vid = torch.load(vid_path, map_location='cpu', mmap=True)
        
        # Pick random frame
        frame_idx = random.randint(0, len(vid) - 1)
        frame = vid[frame_idx]

        return frame  # [c,h,w]

    def __iter__(self):
        while True:
            yield self.get_item()

def collate_fn(x):
    # x is list of frames
    res = torch.stack(x)
    return res  # [b,c,h,w]

def get_loader(batch_size, **data_kwargs):
    """
    Creates a DataLoader for the CoDDataset with the specified batch size
    
    Args:
        batch_size: Number of samples per batch
        **dataloader_kwargs: Additional arguments to pass to DataLoader
        
    Returns:
        DataLoader instance
    """
    dataset = CoDDataset(**data_kwargs)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    return loader

if __name__ == "__main__":
    import time
    loader = get_loader(4)

    start = time.time()
    batch = next(iter(loader))
    end = time.time()

    print(f"Time to load batch: {end-start:.2f}s")
    print(f"Video shape: {batch.shape}")