import torch
import os
import json
from tqdm import tqdm


def clean_and_rebuild_metadata(preprocessed_dir):
    """
    Remove corrupted .pt files and rebuild metadata.json
    
    Args:
        preprocessed_dir: Directory containing preprocessed clips
    """
    print(f"\n{'='*70}")
    print(f"CLEANING AND REBUILDING: {preprocessed_dir}")
    print(f"{'='*70}")
    
    # Get all .pt files
    all_files = sorted([f for f in os.listdir(preprocessed_dir) if f.endswith('.pt')])
    print(f"\nFound {len(all_files)} .pt files")
    
    valid_clips = []
    corrupted_files = []
    
    print("\nScanning files for corruption...")
    for filename in tqdm(all_files, desc="Validating clips"):
        filepath = os.path.join(preprocessed_dir, filename)
        
        try:
            # Check file size first (quick check)
            if os.path.getsize(filepath) == 0:
                corrupted_files.append(filepath)
                continue
            
            # Try to load the file
            data = torch.load(filepath, map_location='cpu')
            
            # Verify it has the required keys
            if 'pixel_values' not in data or 'label' not in data:
                print(f"\nWarning: Missing keys in {filename}")
                corrupted_files.append(filepath)
                continue
            
            # Verify tensor shape
            if data['pixel_values'].shape != torch.Size([16, 3, 224, 224]):
                print(f"\nWarning: Wrong shape in {filename}: {data['pixel_values'].shape}")
                corrupted_files.append(filepath)
                continue
            
            # Extract index from filename (clip_00012345.pt -> 12345)
            index = int(filename.replace('clip_', '').replace('.pt', ''))
            
            # File is valid!
            valid_clips.append({
                'clip_file': filename,
                'label': int(data['label']),
                'index': index
            })
            
        except Exception as e:
            # File is corrupted
            corrupted_files.append(filepath)
    
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"  Valid clips: {len(valid_clips)}")
    print(f"  Corrupted files: {len(corrupted_files)}")
    print(f"{'='*70}")
    
    # Delete corrupted files
    if corrupted_files:
        print(f"\nDeleting {len(corrupted_files)} corrupted files...")
        for filepath in tqdm(corrupted_files, desc="Deleting"):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"\nError deleting {filepath}: {e}")
        print("✓ Corrupted files deleted")
    
    # Sort by index
    valid_clips.sort(key=lambda x: x['index'])
    
    # Print statistics
    if valid_clips:
        labels = [clip['label'] for clip in valid_clips]
        normal = sum(1 for l in labels if l == 0)
        anomaly = sum(1 for l in labels if l == 1)
        
        print(f"\n{'='*70}")
        print(f"FINAL STATISTICS:")
        print(f"  Total clips: {len(valid_clips)}")
        print(f"  Normal: {normal} ({normal/len(labels)*100:.1f}%)")
        print(f"  Anomaly: {anomaly} ({anomaly/len(labels)*100:.1f}%)")
        print(f"  Index range: {valid_clips[0]['index']} to {valid_clips[-1]['index']}")
        print(f"{'='*70}")
    
    # Save metadata
    metadata_path = os.path.join(preprocessed_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(valid_clips, f, indent=2)
    
    print(f"\n✓ Metadata saved to: {metadata_path}")
    print(f"✓ Total valid clips: {len(valid_clips)}")
    
    return valid_clips


# Main function to clean both train and val
def main():
    import sys
    
    # Paths
    TRAIN_DIR = "/home/public/mkamal/datasets/deep_learning/projdata/uploaded_data/preprocessed_clips/train"
    VAL_DIR = "/home/public/mkamal/datasets/deep_learning/projdata/uploaded_data/preprocessed_clips/validation"
    
    print("="*70)
    print("CORRUPTED FILE CLEANUP AND METADATA REBUILD")
    print("="*70)
    
    # Clean training clips
    if os.path.exists(TRAIN_DIR):
        print("\n\nCLEANING TRAINING CLIPS")
        train_clips = clean_and_rebuild_metadata(TRAIN_DIR)
    else:
        print(f"\nWarning: Training directory not found: {TRAIN_DIR}")
        train_clips = []
    
    # # Clean validation clips
    # if os.path.exists(VAL_DIR):
    #     print("\n\nCLEANING VALIDATION CLIPS")
    #     val_clips = clean_and_rebuild_metadata(VAL_DIR)
    # else:
    #     print(f"\nWarning: Validation directory not found: {VAL_DIR}")
    #     val_clips = []
    
    # Summary
    # print("\n" + "="*70)
    # print("COMPLETE SUMMARY")
    # print("="*70)
    # print(f"Training clips: {len(train_clips)}")
    # print(f"Validation clips: {len(val_clips)}")
    # print(f"Total clips: {len(train_clips) + len(val_clips)}")
    # print("="*70)
    
    print("\n✓ Cleanup complete!")
    print("\nYou can now use:")
    print("  train_ds = DoTAVideoMAEDatasetPreprocessed('./preprocessed_clips/train')")
    print("  val_ds = DoTAVideoMAEDatasetPreprocessed('./preprocessed_clips/validation')")


if __name__ == "__main__":
    main()