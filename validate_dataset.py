import os
import glob
import argparse
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
import sys

# Import LangMamba dataset loader
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from data.mayo16_dataset import Mayo16Dataset

def extract_patient_id(filename):
    # Depending on format: 'train_quarter-L067_000.npy' -> split('-')[-1] -> 'L067_000.npy' -> split('_')[0] -> 'L067'
    suffix = filename.split('-')[-1]
    return suffix.split('_')[0]

def extract_suffix(filename):
    return filename.split('-')[-1]

def main():
    print("============================================================")
    print("        LANGMAMBA DATASET VALIDATION")
    print("============================================================")

    dest_dir = Path("datasets/Mayo2016_2d")
    
    # 1. Dataset Structure
    train_q_files = list((dest_dir / "train" / "quarter_1mm").glob("*.npy"))
    train_f_files = list((dest_dir / "train" / "full_1mm").glob("*.npy"))
    test_q_files = list((dest_dir / "test" / "quarter_1mm").glob("*.npy"))
    test_f_files = list((dest_dir / "test" / "full_1mm").glob("*.npy"))
    
    print("\n[1] DATASET STRUCTURE")
    print(f"Train quarter-dose files : {len(train_q_files)}")
    print(f"Train full-dose files    : {len(train_f_files)}")
    print(f"Test quarter-dose files  : {len(test_q_files)}")
    print(f"Test full-dose files     : {len(test_f_files)}")

    # 2. Pair Validation
    train_q_suffixes = [extract_suffix(f.name) for f in train_q_files]
    train_f_suffixes = [extract_suffix(f.name) for f in train_f_files]
    test_q_suffixes = [extract_suffix(f.name) for f in test_q_files]
    test_f_suffixes = [extract_suffix(f.name) for f in test_f_files]
    
    train_matched = len(set(train_q_suffixes).intersection(train_f_suffixes))
    test_matched = len(set(test_q_suffixes).intersection(test_f_suffixes))
    
    missing_pairs = abs(len(train_q_suffixes) - len(train_f_suffixes)) + abs(len(test_q_suffixes) - len(test_f_suffixes))
    
    # Check duplicates by comparing list length to set length
    duplicate_pairs = (len(train_q_suffixes) - len(set(train_q_suffixes))) + (len(train_f_suffixes) - len(set(train_f_suffixes)))
    duplicate_pairs += (len(test_q_suffixes) - len(set(test_q_suffixes))) + (len(test_f_suffixes) - len(set(test_f_suffixes)))

    print("\n[2] PAIR VALIDATION")
    print(f"Matched training pairs   : {train_matched}")
    print(f"Matched testing pairs    : {test_matched}")
    print(f"Missing pairs            : {missing_pairs}")
    print(f"Duplicate pairs          : {duplicate_pairs}")

    # 3. Patient Split
    train_patients = set(extract_patient_id(f.name) for f in train_q_files)
    test_patients = set(extract_patient_id(f.name) for f in test_q_files)
    overlap = len(train_patients.intersection(test_patients))

    print("\n[3] PATIENT SPLIT")
    print(f"Training patients        : {len(train_patients)} {sorted(list(train_patients))}")
    print(f"Testing patients         : {len(test_patients)} {sorted(list(test_patients))}")
    print(f"Patient overlap          : {overlap}")

    # 4. DataLoader Test
    print("\n[4] DATALOADER TEST")
    
    # Mock opt for LangMamba Dataset
    opt = argparse.Namespace()
    opt.dataroot = './datasets/Mayo2016_2d'
    opt.batch_size = 4
    opt.serial_batches = False
    opt.num_threads = 0
    opt.max_dataset_size = float("inf")
    opt.crop_size = 512
    opt.load_size = 512
    opt.no_flip = True

    try:
        opt.phase = 'train'
        opt.isTrain = True
        train_dataset = Mayo16Dataset(opt)
        
        opt.phase = 'test'
        opt.isTrain = False
        test_dataset = Mayo16Dataset(opt)
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
        print(f"Train dataset size       : {len(train_dataset)}")
        print(f"Test dataset size        : {len(test_dataset)}")
        print(f"Batch size               : 4")
        
        # Load one train batch
        train_batch = next(iter(train_loader))
        train_A = train_batch['A']
        train_B = train_batch['B']
        
        print(f"\nQuarter-dose shape       : {train_A.shape}")
        print(f"Full-dose shape          : {train_B.shape}")
        print(f"Quarter-dose dtype       : {train_A.dtype}")
        print(f"Full-dose dtype          : {train_B.dtype}")
        
        print(f"\nQuarter-dose min/max     : {train_A.min().item():.4f} / {train_A.max().item():.4f}")
        print(f"Full-dose min/max        : {train_B.min().item():.4f} / {train_B.max().item():.4f}")
        
        dataloader_pass = "PASS"
        pair_alignment_pass = "PASS" if train_A.shape == train_B.shape and len(train_dataset) > 0 else "FAIL"
        
    except Exception as e:
        print(f"\nError initializing DataLoader: {e}")
        dataloader_pass = "FAIL"
        pair_alignment_pass = "FAIL"

    dataset_pass = "PASS" if missing_pairs == 0 and duplicate_pairs == 0 else "FAIL"
    split_pass = "PASS" if overlap == 0 else "FAIL"
    
    print("\n[5] VALIDATION RESULT")
    print(f"Dataset validation       : {dataset_pass}")
    print(f"DataLoader validation    : {dataloader_pass}")
    print(f"Pair alignment           : {pair_alignment_pass}")
    print(f"Patient split            : {split_pass}")

    print("\n============================================================")
    if all(x == "PASS" for x in [dataset_pass, dataloader_pass, pair_alignment_pass, split_pass]):
        print("             ALL CHECKS PASSED")
    else:
        print("             VALIDATION FAILED")
        sys.exit(1)
    print("============================================================")

if __name__ == "__main__":
    main()
