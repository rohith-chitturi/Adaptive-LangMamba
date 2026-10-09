import os
import glob
import json
import argparse
from pathlib import Path
import shutil

def get_args():
    parser = argparse.ArgumentParser(description="Prepare Mayo 2016 Dataset")
    parser.add_argument("--src", type=str, default="mayo_2016_npy", help="Source directory")
    parser.add_argument("--dest", type=str, default="datasets/Mayo2016_2d", help="Destination directory")
    parser.add_argument("--test-patients", type=str, nargs="+", default=["L506", "L333"], help="Patients to use for testing")
    parser.add_argument("--dry-run", action="store_true", help="Print summary and manifest without copying files")
    return parser.parse_args()

def main():
    args = get_args()
    src_dir = Path(args.src)
    dest_dir = Path(args.dest)
    
    if not src_dir.exists():
        print(f"Error: Source directory {src_dir} does not exist.")
        return

    # Gather all npy files
    npy_files = list(src_dir.glob("*.npy"))
    
    # Dictionary to hold patient -> slice -> {25: path, target: path}
    patients_data = {}
    
    for f in npy_files:
        basename = f.name
        # Expected: L067_25_000_img.npy
        parts = basename.replace("_img.npy", "").split("_")
        if len(parts) >= 3:
            patient_id = parts[0]
            dose = parts[1]
            slice_idx = parts[2]
            
            if patient_id not in patients_data:
                patients_data[patient_id] = {}
            if slice_idx not in patients_data[patient_id]:
                patients_data[patient_id][slice_idx] = {}
                
            patients_data[patient_id][slice_idx][dose] = f

    manifest = []
    
    summary = {
        "train": {"quarter": 0, "full": 0, "patients": []},
        "test": {"quarter": 0, "full": 0, "patients": []}
    }
    
    # Process patients
    for patient_id, slices in patients_data.items():
        split = "test" if patient_id in args.test_patients else "train"
        summary[split]["patients"].append(patient_id)
        
        for slice_idx, doses in slices.items():
            if "25" in doses and "target" in doses:
                q_src = doses["25"]
                f_src = doses["target"]
                
                # Format required by mayo16_dataset.py:
                # self.f_path_list[index].split('-')[-1] == self.q_path_list[index].split('-')[-1]
                # So we use `{split}_quarter-{patient_id}_{slice_idx}.npy`
                
                match_str = f"{patient_id}_{slice_idx}.npy"
                q_dest = dest_dir / split / "quarter_1mm" / f"{split}_quarter-{match_str}"
                f_dest = dest_dir / split / "full_1mm" / f"{split}_full-{match_str}"
                
                manifest.append({
                    "patient_id": patient_id,
                    "slice_idx": slice_idx,
                    "split": split,
                    "quarter_src": q_src.name,
                    "full_src": f_src.name,
                    "quarter_dest": str(q_dest),
                    "full_dest": str(f_dest)
                })
                
                summary[split]["quarter"] += 1
                summary[split]["full"] += 1
                
    print("--- Dry Run Summary ---" if args.dry_run else "--- Execution Summary ---")
    print(f"Total Patients Found: {len(patients_data)}")
    print(f"Train Patients ({len(summary['train']['patients'])}): {summary['train']['patients']}")
    print(f"Test Patients ({len(summary['test']['patients'])}): {summary['test']['patients']}")
    print(f"Train pairs: {summary['train']['quarter']}")
    print(f"Test pairs: {summary['test']['quarter']}")
    print(f"Total pairs to process: {len(manifest)}")
    
    # Dump manifest
    manifest_path = "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
    print(f"\nManifest saved to {manifest_path}")
    
    if args.dry_run:
        print("\nDry run completed. No files were copied.")
        # Show a few manifest examples
        print("Sample manifest entries:")
        for m in manifest[:2]:
            print(f"  {m['quarter_src']} -> {m['quarter_dest']}")
            print(f"  {m['full_src']} -> {m['full_dest']}")
        return

    # Execute Copy
    print("\nCreating directories...")
    for split in ["train", "test"]:
        (dest_dir / split / "quarter_1mm").mkdir(parents=True, exist_ok=True)
        (dest_dir / split / "full_1mm").mkdir(parents=True, exist_ok=True)
        
    print("Copying files (this may take a while)...")
    for m in manifest:
        shutil.copy2(src_dir / m["quarter_src"], m["quarter_dest"])
        shutil.copy2(src_dir / m["full_src"], m["full_dest"])
        
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()
