from torch.utils.data import DataLoader, IterableDataset
import torch
import torch.nn.functional as F

import os
import random

class CoDWFDataset(IterableDataset):
    def __init__(self, window_length = 88200, root = "../cod_data/BlackOpsColdWar"):
        super().__init__()

        self.window_length = window_length
        self.paths = []
        for root_dir in os.listdir(root):
            splits_dir = os.path.join(root, root_dir, "splits")
            if not os.path.isdir(splits_dir):
                continue
                
            # Get all files in splits dir
            files = os.listdir(splits_dir)
            # Filter to just the RGB files
            rgb_files = [f for f in files if f.endswith("_wf.pt")]
            
            for rgb_file in rgb_files:
                rgb_path = os.path.join(splits_dir, rgb_file)
                self.paths.append(rgb_path)
    
    def get_item(self):
        # Pick random audio path
        audio_path = random.choice(self.paths)
        # Load audio tensor with memory mapping
        audio = torch.load(audio_path, map_location='cpu', mmap=True)
        
        # Pick random start index ensuring enough room for window
        max_start_idx = len(audio) - self.window_length
        start_idx = random.randint(0, max_start_idx)
        
        # Get window of audio samples
        samples = audio[start_idx:start_idx + self.window_length]
        
        # Normalize to [-1, 1]
        max_val = samples.abs().max()
        samples = samples / max_val if max_val > 0 else samples
        
        # Random phase flip with 0.5 probability
        if random.random() < 0.5:
            samples = -samples

        return samples.permute(1,0)  # [window_length]

    def __iter__(self):
        while True:
            yield self.get_item()

def collate_fn(x):
    # x is list of frames
    res = torch.stack(x)
    return res  # [b,samples]

def get_loader(batch_size, **data_kwargs):
    """
    Creates a DataLoader for the CoDDataset with the specified batch size
    
    Args:
        batch_size: Number of samples per batch
        **dataloader_kwargs: Additional arguments to pass to DataLoader
        
    Returns:
        DataLoader instance
    """
    dataset = CoDWFDataset(**data_kwargs)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    return loader

if __name__ == "__main__":
    import time
    from tqdm import tqdm
    loader = get_loader(1, root="../cod_download/raw")
    
    n_batches = 1000
    n_with_nans = 0
    
    start = time.time()

    loader = iter(loader)
    for i in tqdm(range(n_batches)):
        batch = next(loader)
        if torch.isnan(batch).any():
            n_with_nans += 1
            
    end = time.time()
    
    print(f"Time to check {n_batches} batches: {end-start:.2f}s")
    print(f"Proportion of batches with NaNs: {n_with_nans/n_batches:.3%}")
    print(f"Number of batches with NaNs: {n_with_nans}/{n_batches}")