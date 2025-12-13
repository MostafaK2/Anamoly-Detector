import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import os
import json
from torchvision import transforms
import cv2

from ConvAE.ConvAE import Conv2DAutoEncoder 
from utils import read_json_files
from ConvAE.ConvAEDatasetClasses import StackedFramesDatasetTest

# ==================== CONFIGURATION ====================
MODEL_PATH = "/home/public/mkamal/saved_models/convAE_best_model30epochs3over.pth"
TEST_TXT_PATH = "test.txt"
ROOT_DIR = "/home/public/mkamal/datasets/deep_learning/projdata/uploaded_data"
OUTPUT_DIR = "high_error_visualizations"

# Image settings
IMAGE_SIZE = 224
NORM_MEAN = [0.4333332]
NORM_STD = [0.2551518]
CONVERT_GRAY = 1
IMAGE_STACK = 10

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Visualization settings
NUM_VIDEOS_TO_VISUALIZE = 10  # Top 10 highest errors
FRAMES_PER_VIDEO = 5  # Show 5 frames from each stack

print("="*60)
print("VISUALIZING HIGH RECONSTRUCTION ERROR VIDEOS")
print("="*60 + "\n")

# ==================== LOAD MODEL ====================
print("Loading model...")
model = Conv2DAutoEncoder(CONVERT_GRAY * IMAGE_STACK).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
print("✓ Model loaded\n")

# ==================== IMAGE TRANSFORM ====================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=CONVERT_GRAY),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

# For visualization (without normalization)
viz_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=CONVERT_GRAY),
    transforms.ToTensor()
])

# ==================== CREATE DATASET ====================
print("Creating test dataset...")
json_list = read_json_files(TEST_TXT_PATH)
test_db = StackedFramesDatasetTest(ROOT_DIR, json_list, transform=transform, only_normal=False)

# Create DataLoader
test_loader = DataLoader(test_db, batch_size=1, shuffle=False, num_workers=4)
print(f"✓ Dataset created with {len(test_db)} stacks\n")

# ==================== COMPUTE ERRORS FOR ALL STACKS ====================
print("Computing reconstruction errors for all stacks...")
all_errors = []
all_labels = []
all_stacks = []
all_reconstructions = []

with torch.no_grad():
    for batch_frames, batch_labels in tqdm(test_loader, desc="Processing"):
        batch_frames = batch_frames.to(DEVICE)
        reconstruction = model(batch_frames)
        mse = torch.mean((batch_frames - reconstruction) ** 2).item()
        
        all_errors.append(mse)
        all_labels.append(batch_labels.item())
        all_stacks.append(batch_frames.cpu())
        all_reconstructions.append(reconstruction.cpu())

all_errors = np.array(all_errors)
all_labels = np.array(all_labels)

print(f"✓ Computed errors for {len(all_errors)} stacks\n")

# ==================== FIND TOP HIGH ERROR STACKS ====================
# Get indices of top N highest errors
top_error_indices = np.argsort(all_errors)[-NUM_VIDEOS_TO_VISUALIZE:][::-1]

print(f"Top {NUM_VIDEOS_TO_VISUALIZE} highest reconstruction errors:")
for i, idx in enumerate(top_error_indices):
    label_str = "Anomalous" if all_labels[idx] == 1 else "Normal"
    print(f"  {i+1}. Stack {idx}: Error={all_errors[idx]:.6f}, Label={label_str}")
print()

# ==================== CREATE OUTPUT DIRECTORY ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== DENORMALIZATION FUNCTION ====================
def denormalize(tensor, mean, std):
    """Denormalize tensor for visualization"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# ==================== VISUALIZE EACH HIGH ERROR STACK ====================
print("Generating visualizations...")

for rank, idx in enumerate(top_error_indices):
    print(f"Visualizing stack {rank+1}/{NUM_VIDEOS_TO_VISUALIZE}...")
    
    stack = all_stacks[idx]  # [1, 10, 224, 224]
    reconstruction = all_reconstructions[idx]  # [1, 10, 224, 224]
    error = all_errors[idx]
    label = all_labels[idx]
    label_str = "Anomalous" if label == 1 else "Normal"
    
    # Denormalize
    stack_denorm = denormalize(stack.clone().squeeze(0), NORM_MEAN, NORM_STD)
    reconstruct_denorm = denormalize(reconstruction.clone().squeeze(0), NORM_MEAN, NORM_STD)
    
    # Compute per-frame error map
    error_map = (stack_denorm - reconstruct_denorm) ** 2
    
    # Select frames to display (evenly spaced)
    frame_indices = np.linspace(0, IMAGE_STACK-1, FRAMES_PER_VIDEO, dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(3, FRAMES_PER_VIDEO, figsize=(3*FRAMES_PER_VIDEO, 12))
    
    for col_idx, frame_idx in enumerate(frame_indices):
        # Original frame
        orig = stack_denorm[frame_idx].numpy()
        axes[0, col_idx].imshow(orig, cmap='gray', vmin=0, vmax=1)
        axes[0, col_idx].set_title(f'Frame {frame_idx}', fontsize=10)
        axes[0, col_idx].axis('off')
        
        # Error map
        error_frame = error_map[frame_idx].numpy()
        im = axes[1, col_idx].imshow(error_frame, cmap='hot', vmin=0, vmax=error_frame.max())
        axes[1, col_idx].set_title('Error Map', fontsize=10)
        axes[1, col_idx].axis('off')
        
        # Difference (enhanced)
        recon = reconstruct_denorm[frame_idx].numpy()
        diff = np.abs(orig - recon)**2
        axes[2, col_idx].imshow(diff, cmap='hot', vmin=0, vmax=diff.max())
        axes[2, col_idx].set_title('Abs Difference', fontsize=10)
        axes[2, col_idx].axis('off')
    
    # Add row labels
    axes[0, 0].text(-0.1, 0.5, 'Original', transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold', va='center', rotation=90)
    axes[1, 0].text(-0.1, 0.5, 'Error Map', transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold', va='center', rotation=90)
    axes[2, 0].text(-0.1, 0.5, 'Abs Difference', transform=axes[2, 0].transAxes, fontsize=12, fontweight='bold', va='center', rotation=90)
    # Main title
    fig.suptitle(f'Rank #{rank+1} - Stack {idx}\n'f'Reconstruction Error: {error:.6f} | Label: {label_str}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(OUTPUT_DIR, f'high_error_rank{rank+1}_stack{idx}_{label_str}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")

# ==================== CREATE ERROR DISTRIBUTION WITH HIGHLIGHTS ====================
print("\nGenerating error distribution plot...")

fig, ax = plt.subplots(figsize=(12, 6))

# Plot histograms for normal and anomalous
normal_errors = all_errors[all_labels == 0]
anomalous_errors = all_errors[all_labels == 1]

ax.hist(normal_errors, bins=50, alpha=0.6, label=f'Normal (n={len(normal_errors)})', color='green', edgecolor='black')
ax.hist(anomalous_errors, bins=50, alpha=0.6, label=f'Anomalous (n={len(anomalous_errors)})', color='red', edgecolor='black')

# Highlight the top error stacks
for rank, idx in enumerate(top_error_indices):
    color = 'red' if all_labels[idx] == 1 else 'green'
    marker = 'x' if all_labels[idx] == 1 else 'o'
    ax.axvline(all_errors[idx], color=color, linestyle='--', alpha=0.7, linewidth=2)
    ax.plot(all_errors[idx], 0, marker=marker, markersize=10, color=color, 
            label=f'Rank {rank+1}' if rank < 3 else '')

ax.set_xlabel('Reconstruction Error', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Reconstruction Error Distribution with Top High-Error Stacks Highlighted', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
dist_plot_path = os.path.join(OUTPUT_DIR, 'error_distribution_highlighted.png')
plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {dist_plot_path}")

# ==================== CREATE SUMMARY STATISTICS ====================
print("\nGenerating summary statistics...")

summary_path = os.path.join(OUTPUT_DIR, 'high_error_summary.txt')
with open(summary_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("HIGH RECONSTRUCTION ERROR STACKS - SUMMARY\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Total stacks analyzed: {len(all_errors)}\n")
    f.write(f"Visualizations created: {NUM_VIDEOS_TO_VISUALIZE}\n\n")
    
    f.write("TOP HIGH ERROR STACKS:\n")
    f.write("-"*60 + "\n")
    
    for rank, idx in enumerate(top_error_indices):
        label_str = "Anomalous (1)" if all_labels[idx] == 1 else "Normal (0)"
        f.write(f"Rank {rank+1}:\n")
        f.write(f"  Stack Index: {idx}\n")
        f.write(f"  Reconstruction Error: {all_errors[idx]:.6f}\n")
        f.write(f"  Ground Truth Label: {label_str}\n")
        f.write(f"  Percentile: {(np.sum(all_errors <= all_errors[idx]) / len(all_errors) * 100):.2f}%\n")
        f.write("\n")
    
    # Statistics on high error stacks
    high_error_labels = [all_labels[idx] for idx in top_error_indices]
    num_normal_high = sum(1 for l in high_error_labels if l == 0)
    num_anomalous_high = sum(1 for l in high_error_labels if l == 1)
    
    f.write("LABEL DISTRIBUTION IN TOP HIGH ERRORS:\n")
    f.write("-"*60 + "\n")
    f.write(f"Normal stacks: {num_normal_high} ({num_normal_high/len(high_error_labels)*100:.1f}%)\n")
    f.write(f"Anomalous stacks: {num_anomalous_high} ({num_anomalous_high/len(high_error_labels)*100:.1f}%)\n\n")
    
    f.write("\n" + "="*60 + "\n")

print(f"✓ Saved: {summary_path}")

# ==================== FINAL SUMMARY ====================
print("\n" + "="*60)
print("VISUALIZATION COMPLETE!")
print("="*60)
print(f"Output directory: {OUTPUT_DIR}/")
print(f"  - {NUM_VIDEOS_TO_VISUALIZE} visualization images")
print(f"  - error_distribution_highlighted.png")
print(f"  - high_error_summary.txt")
print("\nFiles saved:")
for rank in range(NUM_VIDEOS_TO_VISUALIZE):
    idx = top_error_indices[rank]
    label_str = "Anomalous" if all_labels[idx] == 1 else "Normal"
    print(f"  Rank {rank+1}: high_error_rank{rank+1}_stack{idx}_{label_str}.png")
print("="*60)