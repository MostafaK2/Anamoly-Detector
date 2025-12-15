import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from PIL import Image
from tqdm import tqdm
from transformers import VideoMAEImageProcessor
from utils import split_before_training


def preprocess_and_save_clips(root_dir, file_list, processor, sequence_size=16, overlap=0, output_dir="train", start_from=0):
    os.makedirs(output_dir, exist_ok=True)
    
    # First, create all sequences
    sequences = []
    annotations_path = os.path.join(root_dir, "annotations")
    
    print("Building sequences from JSON files")
    for file in tqdm(file_list, desc="Processing JSON files"):
        json_path = os.path.join(annotations_path, file)
        if not os.path.exists(json_path):
            continue
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        frames_data = []
        for label_info in data["labels"]:
            accident_name = label_info.get("accident_name", "normal")
            image_path = label_info.get("image_path", "")
            if not image_path:
                continue
            
            full_path = os.path.join(root_dir, image_path)
            label = 0 if accident_name == "normal" else 1
            
            frames_data.append({
                'path': full_path,
                'label': label,
                'accident_name': accident_name
            })
        
        # Generate sequences
        sliding_skip = sequence_size - overlap if overlap > 0 else sequence_size
        for i in range(0, len(frames_data), sliding_skip):
            sequence_frames = frames_data[i:i + sequence_size]
            
            if len(sequence_frames) < sequence_size:
                continue
            
            frame_paths = [f['path'] for f in sequence_frames]
            sequence_labels = [f['label'] for f in sequence_frames]
            sequence_label = 1 if any(sequence_labels) else 0
            
            sequences.append({
                'paths': frame_paths,
                'label': sequence_label
            })
    
    print(f"\nTotal sequences to preprocess: {len(sequences)}")
    
    # Statistics
    if sequences:
        labels = [s['label'] for s in sequences]
        normal = sum(1 for l in labels if l == 0)
        anomaly = sum(1 for l in labels if l == 1)
        print(f"  Normal: {normal} ({normal/len(labels)*100:.1f}%)")
        print(f"  Anomaly: {anomaly} ({anomaly/len(labels)*100:.1f}%)")
    else:
        print("ERROR: No sequences found!")
        return []
    
    # Build complete metadata from existing files
    metadata = []
    metadata_path = os.path.join(output_dir, 'metadata.json')
    
    print("\nScanning for existing preprocessed clips")
    for idx in tqdm(range(len(sequences))):
        clip_filename = f"clip_{idx:08d}.pt"
        clip_path = os.path.join(output_dir, clip_filename)
        
        if os.path.exists(clip_path):
            try:
                # Load to get label
                data = torch.load(clip_path, map_location='cpu')
                metadata.append({
                    'clip_file': clip_filename,
                    'label': int(data['label']),
                    'index': idx
                })
            except Exception as e:
                print(f"Warning: Corrupted file {clip_path}: {e}")
    
    print(f"Found {len(metadata)} existing clips")
    
    # Now preprocess remaining sequences
    failed_count = 0
    
    for idx in tqdm(range(start_from, len(sequences)), desc="Saving clips", initial=start_from, total=len(sequences)):
        sequence = sequences[idx]
        frame_paths = sequence['paths']
        label = sequence['label']
        
        clip_filename = f"clip_{idx:08d}.pt"
        clip_path = os.path.join(output_dir, clip_filename)
        
        # Skip if already exists and valid
        if os.path.exists(clip_path):
            try:
                if os.path.getsize(clip_path) > 0:
                    continue  # Already done
                else:
                    os.remove(clip_path)  # Remove empty file
            except:
                pass
        
        # Load frames
        frames = []
        valid = True
        for img_path in frame_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                frames.append(img)
            except Exception as e:
                print(f"\nError loading {img_path}: {e}")
                valid = False
                break
        
        if not valid or len(frames) != sequence_size:
            failed_count += 1
            continue
        
        try:
            # Process with VideoMAE processor
            inputs = processor(frames, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)  # [16, 3, 224, 224]
            
            # Save with retry
            max_retries = 3
            saved = False
            for retry in range(max_retries):
                try:
                    torch.save({
                        'pixel_values': pixel_values,
                        'label': label
                    }, clip_path)
                    
                    # Verify
                    if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                        saved = True
                        break
                except Exception as e:
                    print(f"\nSave error (attempt {retry+1}): {e}")
                    if retry < max_retries - 1:
                        import time
                        time.sleep(0.5)
            
            if saved:
                # Add to metadata (remove old entry if exists)
                metadata = [m for m in metadata if m['index'] != idx]
                metadata.append({
                    'clip_file': clip_filename,
                    'label': label,
                    'index': idx
                })
                
                # Save metadata periodically every 1000 clips
                if len(metadata) % 1000 == 0:
                    temp_metadata = sorted(metadata, key=lambda x: x['index'])
                    with open(metadata_path, 'w') as f:
                        json.dump(temp_metadata, f, indent=2)
                    print(f"\n  Checkpoint: Saved metadata ({len(temp_metadata)} clips)")
            else:
                failed_count += 1
        
        except Exception as e:
            print(f"\nError processing clip {idx}: {e}")
            failed_count += 1
    
    # CRITICAL: Always save final complete metadata
    print("\nFinalizing metadata...")
    
    # Build complete metadata by scanning ALL files
    final_metadata = []
    for idx in range(len(sequences)):
        clip_filename = f"clip_{idx:08d}.pt"
        clip_path = os.path.join(output_dir, clip_filename)
        
        if os.path.exists(clip_path):
            try:
                if os.path.getsize(clip_path) > 0:
                    # Load to verify and get label
                    data = torch.load(clip_path, map_location='cpu')
                    final_metadata.append({
                        'clip_file': clip_filename,
                        'label': int(data['label']),
                        'index': idx
                    })
            except Exception as e:
                print(f"Warning: Skipping corrupted file {clip_path}")
    
    # Sort by index
    final_metadata.sort(key=lambda x: x['index'])
    
    # Save final metadata
    with open(metadata_path, 'w') as f:
        json.dump(final_metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ Total clips saved: {len(final_metadata)}")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Metadata saved: {metadata_path}")
    if failed_count > 0:
        print(f"⚠ Failed sequences: {failed_count}")
    
    # Print statistics
    if final_metadata:
        labels = [m['label'] for m in final_metadata]
        normal = sum(1 for l in labels if l == 0)
        anomaly = sum(1 for l in labels if l == 1)
        print(f"\nFinal Statistics:")
        print(f"  Normal: {normal} ({normal/len(labels)*100:.1f}%)")
        print(f"  Anomaly: {anomaly} ({anomaly/len(labels)*100:.1f}%)")
    print(f"{'='*70}")
    
    return final_metadata


def main():
    # Load processor
    processor = VideoMAEImageProcessor.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics"
    )
    
    # Modify these
    SEED = 42
    ROOT = "/home/public/mkamal/datasets/deep_learning/projdata/uploaded_data"
    JSON_FILE_PATH = "/home/grad/masters/2025/mkamal/mkamal/dl_project/anamoly-detection/train_val.txt"
    
    # Split data
    train_file_list, val_file_list = split_before_training(JSON_FILE_PATH, 0.1, SEED)
    
    print(f"Train files: {len(train_file_list)}")
    print(f"Val files: {len(val_file_list)}")
    
    # Preprocess TRAINING clips
   
    # preprocess_and_save_clips(
    #     root_dir=ROOT,
    #     file_list=train_file_list,
    #     processor=processor,
    #     sequence_size=16,
    #     overlap=0,
    #     output_dir=ROOT + "/preprocessed_clips/train",
    #     start_from=0  # Will auto-skip existing files
    # )
    
    # # Preprocess VALIDATION clips
    # print("\n" + "="*70)
    # print("PREPROCESSING VALIDATION CLIPS")
    # print("="*70)
    # preprocess_and_save_clips(
    #     root_dir=ROOT,
    #     file_list=val_file_list,
    #     processor=processor,
    #     sequence_size=16,
    #     overlap=0,
    #     output_dir=ROOT + "/preprocessed_clips/validation",
    #     start_from=0
    # )


if __name__ == "__main__":
    main()