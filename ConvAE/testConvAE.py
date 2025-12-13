# Standard library
import os
import json
from pathlib import Path

# Third-party
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image

# Local modules
from ConvAE.ConvAE import Conv2DAutoEncoder
from utils import read_json_files
from ConvAE.ConvAEDatasetClasses import StackedFramesDatasetTest



class StackedFramesDatasetTest(Dataset):
    """
    Dataset that returns 10 consecutive normal frames stacked along channel dimension.
    
    """
    def __init__(self, root_dir, file_list, json_file_path = None, stack_size=10, overlap=0, transform=None, only_normal=False):
        
        assert 0 <= overlap < stack_size, "Overlap must be smaller than stack size."

        # Read JSON filenames from the text file
        if (json_file_path):
            assert os.path.exists(json_file_path), f"JSON list file not found: {json_file_path}"
            json_files = read_json_files(json_file_path)
            print("Imported files from .txt file")
        else:
            json_files = file_list

        _sliding_skip = stack_size-overlap

        self.root_dir = root_dir
        self.transform = transform
        self.only_normal = only_normal
        self.stack_size = stack_size
        self.stacks = []  # list of tuples: (video_dir, list of consecutive normal frame paths)
        self.stack_labels = []

        annotations_path = os.path.join(root_dir, "annotations")
        for json_file in json_files:
                json_path = os.path.join(annotations_path, json_file)
                if not os.path.exists(json_path):
                    print(json_path)
                    print("JSON file doesnt exist")
                    continue
            
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    frames = []
                    frame_indices = []

                    anomaly_start = data.get("anomaly_start", None)
                    anomaly_end = data.get("anomaly_end", None)

                    for label in data["labels"]:
                        if only_normal and label["accident_name"] != "normal": # Skips anamolous frames
                            continue

                        frame_idx = label["frame_id"]
                        frames.append(os.path.join(root_dir, label["image_path"]))
                        frame_indices.append(frame_idx)
                    
                    # generate stacks of consecutive frames
                    # Generate stacks of consecutive frames
                    for i in range(0, len(frames), _sliding_skip):
                        stack = frames[i:i + stack_size]
                        stack_frame_indices = frame_indices[i:i + stack_size]
                        
                        # Skip last incomplete stack
                        if len(stack) != stack_size:
                            continue
                        
                        # Determine if this stack contains anomalous frames
                        is_anomalous = False
                        
                        if anomaly_start is not None and anomaly_end is not None:
                            # Check if ANY frame in the stack falls within anomaly range
                            for frame_idx in stack_frame_indices:
                                if anomaly_start <= frame_idx <= anomaly_end:
                                    is_anomalous = True
                                    break
                        
                        # If only_normal is True, skip anomalous stacks
                        if only_normal and is_anomalous:
                            continue
                        
                        # Add stack and its label
                        self.stacks.append(stack)
                        self.stack_labels.append(1 if is_anomalous else 0)
            
        print(f"Generated {len(self.stacks)} stacks of 10 frame clips, where the label distribution is: Normal={sum(1 for l in self.stack_labels if l == 0)}, Anomalous={sum(1 for l in self.stack_labels if l == 1)}")

    def __len__(self):
        return len(self.stacks)

    def __getitem__(self, idx):
        stack_paths = self.stacks[idx]
        frames = []
        for img_path in stack_paths:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        # stack along channel dimension
        stacked_tensor = torch.cat(frames, dim=0)
        return stacked_tensor, self.stack_labels[idx]



# ==================== CONFIGURATION ====================
MODEL_PATH = "/home/public/mkamal/saved_models/convAE_best_model30epochs3over.pth"
TEST_TXT_PATH = "test.txt"  # File containing JSON filenames for test videos
ROOT_DIR = "/home/public/mkamal/datasets/deep_learning/projdata/uploaded_data"
OUTPUT_DIR = "evaluation_results"

# Image settings (from your config)
IMAGE_SIZE = 224
NORM_MEAN = [0.4333332]
NORM_STD = [0.2551518]
CONVERT_GRAY = 1
IMAGE_STACK = 10  # Stack 10 frames

BATCH_SIZE = 32  # Adjust based on your GPU memory
NUM_WORKERS = 8

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*60)
print("VIDEO ANOMALY DETECTION - EVALUATION")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")
print(f"Test file: {TEST_TXT_PATH}")
print(f"Root dir: {ROOT_DIR}")
print("="*60 + "\n")

# ==================== LOAD MODEL ====================
print("Loading model...")
model = Conv2DAutoEncoder(CONVERT_GRAY * IMAGE_STACK).to(DEVICE)

# Load the saved state dict
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

print(f"✓ Model loaded successfully")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# ==================== IMAGE TRANSFORM ====================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=CONVERT_GRAY),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

# ==================== CREATE DATASET ====================
print("Creating test dataset...")
json_list = read_json_files(TEST_TXT_PATH)
test_db = StackedFramesDatasetTest(ROOT_DIR, json_list, transform=transform, only_normal=False)

# Create DataLoader
test_loader = DataLoader(
    test_db, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"\nDataset created with {len(test_db)} stacks\n")

# ==================== EVALUATE ALL STACKS ====================
print("Evaluating all stacks...")
reconstruction_errors = []
regularity_scores = []
all_labels = []

with torch.no_grad():
    for batch_frames, batch_labels in tqdm(test_loader, desc="Processing batches"):
        batch_frames = batch_frames.to(DEVICE)
        
        reconstruction = model(batch_frames) 
        mse = torch.mean((batch_frames - reconstruction) ** 2, dim=(1, 2, 3))
        
        # Store results
        reconstruction_errors.extend(mse.cpu().numpy())
        all_labels.extend(batch_labels.numpy())

# Convert to numpy arrays
reconstruction_errors = np.array(reconstruction_errors)
all_labels = np.array(all_labels)

# Compute regularity scores
regularity_scores = 1.0 / (1.0 + reconstruction_errors)

print(f"\n✓ Evaluation complete!")
print(f"  Processed: {len(all_labels)} stacks")
print(f"  Normal (0): {np.sum(all_labels == 0)}")
print(f"  Anomalous (1): {np.sum(all_labels == 1)}\n")

# ==================== CALCULATE METRICS ====================
print("="*60)
print("METRICS")
print("="*60)

# Reconstruction Error Statistics
print("\nRECONSTRUCTION ERROR:")
print(f"  Overall Mean: {np.mean(reconstruction_errors):.6f}")
print(f"  Overall Std:  {np.std(reconstruction_errors):.6f}")
print(f"  Overall Min:  {np.min(reconstruction_errors):.6f}")
print(f"  Overall Max:  {np.max(reconstruction_errors):.6f}")

normal_errors = reconstruction_errors[all_labels == 0]
anomalous_errors = reconstruction_errors[all_labels == 1]

if len(normal_errors) > 0:
    print(f"\n  Normal stacks (label=0):")
    print(f"    Mean: {np.mean(normal_errors):.6f}")
    print(f"    Std:  {np.std(normal_errors):.6f}")
    print(f"    Min:  {np.min(normal_errors):.6f}")
    print(f"    Max:  {np.max(normal_errors):.6f}")

if len(anomalous_errors) > 0:
    print(f"\n  Anomalous stacks (label=1):")
    print(f"    Mean: {np.mean(anomalous_errors):.6f}")
    print(f"    Std:  {np.std(anomalous_errors):.6f}")
    print(f"    Min:  {np.min(anomalous_errors):.6f}")
    print(f"    Max:  {np.max(anomalous_errors):.6f}")

# Regularity Score Statistics
print("\n" + "-"*60)
print("REGULARITY SCORE:")
print(f"  Overall Mean: {np.mean(regularity_scores):.6f}")
print(f"  Overall Std:  {np.std(regularity_scores):.6f}")
print(f"  Overall Min:  {np.min(regularity_scores):.6f}")
print(f"  Overall Max:  {np.max(regularity_scores):.6f}")

normal_reg = regularity_scores[all_labels == 0]
anomalous_reg = regularity_scores[all_labels == 1]

if len(normal_reg) > 0:
    print(f"\n  Normal stacks (label=0):")
    print(f"    Mean: {np.mean(normal_reg):.6f}")
    print(f"    Std:  {np.std(normal_reg):.6f}")

if len(anomalous_reg) > 0:
    print(f"\n  Anomalous stacks (label=1):")
    print(f"    Mean: {np.mean(anomalous_reg):.6f}")
    print(f"    Std:  {np.std(anomalous_reg):.6f}")

# AUC-ROC and STAUCC
print("\n" + "-"*60)
print("PERFORMANCE METRICS:")
if len(np.unique(all_labels)) >= 2:
    auc_roc = roc_auc_score(all_labels, reconstruction_errors)
    print(f"  AUC-ROC Score: {auc_roc:.4f}")
    
    # STAUCC (for video-level evaluation, same as AUC-ROC)
    staucc = auc_roc
    print(f"  STAUCC Score:  {staucc:.4f}")
else:
    auc_roc = None
    staucc = None
    print("  AUC-ROC: Cannot compute (need both classes)")
    print("  STAUCC: Cannot compute (need both classes)")

print("="*60 + "\n")

# ==================== CREATE OUTPUT DIRECTORY ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== PLOT ROC CURVE ====================
if len(np.unique(all_labels)) >= 2:
    print("Generating ROC curve...")
    fpr, tpr, thresholds = roc_curve(all_labels, reconstruction_errors)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Anomaly Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_path = os.path.join(OUTPUT_DIR, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {roc_path}")

# ==================== PLOT SCORE DISTRIBUTIONS ====================
print("Generating score distributions...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Reconstruction error distribution
if len(normal_errors) > 0:
    axes[0].hist(normal_errors, bins=30, alpha=0.7, 
                label=f'Normal (n={len(normal_errors)})', color='green', edgecolor='black')
if len(anomalous_errors) > 0:
    axes[0].hist(anomalous_errors, bins=30, alpha=0.7, 
                label=f'Anomalous (n={len(anomalous_errors)})', color='red', edgecolor='black')
axes[0].set_xlabel('Reconstruction Error', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Regularity score distribution
if len(normal_reg) > 0:
    axes[1].hist(normal_reg, bins=30, alpha=0.7, 
                label=f'Normal (n={len(normal_reg)})', color='green', edgecolor='black')
if len(anomalous_reg) > 0:
    axes[1].hist(anomalous_reg, bins=30, alpha=0.7, 
                label=f'Anomalous (n={len(anomalous_reg)})', color='red', edgecolor='black')
axes[1].set_xlabel('Regularity Score', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Regularity Score Distribution', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
dist_path = os.path.join(OUTPUT_DIR, 'score_distribution.png')
plt.savefig(dist_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {dist_path}")

# ==================== GENERATE TEXT REPORT ====================
print("\nGenerating evaluation report...")
report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')

with open(report_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("VIDEO ANOMALY DETECTION - EVALUATION REPORT\n")
    f.write("Conv2DAutoEncoder - Stack of 10 Frames\n")
    f.write("="*60 + "\n\n")
    
    f.write("MODEL CONFIGURATION\n")
    f.write("-"*60 + "\n")
    f.write(f"Model Path: {MODEL_PATH}\n")
    f.write(f"Image Stack Size: {IMAGE_STACK}\n")
    f.write(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}\n")
    f.write(f"Grayscale Channels: {CONVERT_GRAY}\n")
    f.write(f"Input Channels: {CONVERT_GRAY * IMAGE_STACK}\n\n")
    
    f.write("DATASET STATISTICS\n")
    f.write("-"*60 + "\n")
    f.write(f"Total stacks: {len(all_labels)}\n")
    f.write(f"Normal stacks (0): {np.sum(all_labels == 0)}\n")
    f.write(f"Anomalous stacks (1): {np.sum(all_labels == 1)}\n\n")
    
    f.write("RECONSTRUCTION ERROR STATISTICS\n")
    f.write("-"*60 + "\n")
    f.write(f"Overall Mean: {np.mean(reconstruction_errors):.6f}\n")
    f.write(f"Overall Std: {np.std(reconstruction_errors):.6f}\n")
    f.write(f"Overall Min: {np.min(reconstruction_errors):.6f}\n")
    f.write(f"Overall Max: {np.max(reconstruction_errors):.6f}\n\n")
    
    if len(normal_errors) > 0:
        f.write(f"Normal stacks (0):\n")
        f.write(f"  Mean: {np.mean(normal_errors):.6f}\n")
        f.write(f"  Std: {np.std(normal_errors):.6f}\n")
        f.write(f"  Min: {np.min(normal_errors):.6f}\n")
        f.write(f"  Max: {np.max(normal_errors):.6f}\n\n")
    
    if len(anomalous_errors) > 0:
        f.write(f"Anomalous stacks (1):\n")
        f.write(f"  Mean: {np.mean(anomalous_errors):.6f}\n")
        f.write(f"  Std: {np.std(anomalous_errors):.6f}\n")
        f.write(f"  Min: {np.min(anomalous_errors):.6f}\n")
        f.write(f"  Max: {np.max(anomalous_errors):.6f}\n\n")
    
    f.write("REGULARITY SCORE STATISTICS\n")
    f.write("-"*60 + "\n")
    f.write(f"Overall Mean: {np.mean(regularity_scores):.6f}\n")
    f.write(f"Overall Std: {np.std(regularity_scores):.6f}\n\n")
    
    if len(normal_reg) > 0:
        f.write(f"Normal stacks (0) - Mean: {np.mean(normal_reg):.6f}, Std: {np.std(normal_reg):.6f}\n")
    
    if len(anomalous_reg) > 0:
        f.write(f"Anomalous stacks (1) - Mean: {np.mean(anomalous_reg):.6f}, Std: {np.std(anomalous_reg):.6f}\n\n")
    
    f.write("PERFORMANCE METRICS\n")
    f.write("-"*60 + "\n")
    if auc_roc is not None:
        f.write(f"AUC-ROC: {auc_roc:.4f}\n")
        f.write(f"STAUCC: {staucc:.4f}\n")
    else:
        f.write("AUC-ROC: Cannot compute (need both classes)\n")
        f.write("STAUCC: Cannot compute (need both classes)\n")
    
    f.write("\n" + "="*60 + "\n")

print(f"✓ Saved: {report_path}")

# ==================== SUMMARY ====================
print("\n" + "="*60)
print("EVALUATION COMPLETE!")
print("="*60)
print(f"Results saved to: {OUTPUT_DIR}/")
print(f"  - roc_curve.png")
print(f"  - score_distribution.png")
print(f"  - evaluation_report.txt")
if auc_roc is not None:
    print(f"\nKey Metrics:")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"  STAUCC:  {staucc:.4f}")
print("="*60)

