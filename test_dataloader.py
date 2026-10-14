import os
import argparse
import torch
import numpy as np
from data.mayo16_dataset import Mayo16Dataset
from torch.utils.data import DataLoader

def main():
    # Create a mock opt object
    opt = argparse.Namespace()
    opt.dataroot = './datasets/Mayo2016_2d'
    opt.phase = 'train'
    opt.isTrain = True
    opt.batch_size = 4
    opt.serial_batches = False
    opt.num_threads = 0
    opt.max_dataset_size = float("inf")
    
    # Also add standard base options if Mayo16Dataset or transforms expect them
    # Some transforms might expect opt.crop_size, opt.load_size etc, let's check
    # actually get_transforms uses none of those, except commented out ones.
    
    print("Testing LangMamba Dataset Loader...")
    try:
        dataset = Mayo16Dataset(opt)
        print(f"Dataset successfully created! Total pairs: {len(dataset)}")
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0
        )
        
        print("Fetching one batch...")
        for i, batch in enumerate(dataloader):
            A = batch['A'] # Quarter dose
            B = batch['B'] # Full dose
            A_paths = batch['A_paths']
            B_paths = batch['B_paths']
            
            print(f"Batch {i} fetched successfully!")
            print(f"  Input A (Quarter) shape: {A.shape}, dtype: {A.dtype}")
            print(f"  Target B (Full) shape: {B.shape}, dtype: {B.dtype}")
            print(f"  A Paths: {[os.path.basename(p) for p in A_paths]}")
            print(f"  B Paths: {[os.path.basename(p) for p in B_paths]}")
            
            # Print value ranges
            print(f"  Input A (Quarter) min: {A.min().item():.3f}, max: {A.max().item():.3f}")
            print(f"  Target B (Full) min: {B.min().item():.3f}, max: {B.max().item():.3f}")
            
            break # Just test one batch
            
        print("\nDataset DataLoader test passed successfully!")
        
    except Exception as e:
        print(f"Error testing dataloader: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
