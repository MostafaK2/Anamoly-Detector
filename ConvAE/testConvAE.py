# Standard library
import os

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm

# PyTorch
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ConvAE_pipeline import read_json_files, Conv2DAutoEncoder,  StackedFramesDatasetTest, Config



# ----------------------------- Testing Configuration -----------------------------
MODEL_PATH = "/home/public/mkamal/saved_models/convAE_best_model30epochs3over.pth"
TEST_TXT_PATH = "/home/grad/masters/2025/mkamal/mkamal/dl_project/anamoly-detection/test.txt"  # File containing JSON filenames for test videos
OUTPUT_DIR = "/home/grad/masters/2025/mkamal/mkamal/dl_project/anamoly-detection/ConvAE/evaluation_results"
BATCH_SIZE = 32  # configure your batch size
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Image settings (from your config)
config = Config()



print("="*60 + "\n ConvAE - Evaluation\n" + "="*60)

# ----------------------------- Load Model -----------------------------
model = Conv2DAutoEncoder(config.CONVERT_GRAY * config.IMAGE_STACK).to(DEVICE)

# Load the saved state dict
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

print(f"Model loaded successfully")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")


# ----------------------------- IMAGE TRANSFORM -----------------------------
transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=config.CONVERT_GRAY),
    transforms.ToTensor(),
    transforms.Normalize(config.NORM_MEAN, config.NORM_STD)
])

# ----------------------------- CREATE DATASET -----------------------------
json_list = read_json_files(TEST_TXT_PATH)
test_db = StackedFramesDatasetTest(config.ROOT, json_list, transform=transform, only_normal=False)

# Create DataLoader
test_loader = DataLoader(
    test_db, 
    batch_size=BATCH_SIZE, # or use ur configurations
    shuffle=False, 
    num_workers=config.NUM_WORKERS,
    pin_memory=True
)

print(f"\nDataset created with {len(test_db)} stacks\n")

# ----------------------------- EVALUATE ALL STACKS -----------------------------
print("Starting evaluation")
reconstruction_errors = []
regularity_scores = []
all_labels = []

with torch.no_grad():
    for batch_frames, batch_labels in tqdm(test_loader, desc="Processing batches"):
        batch_frames = batch_frames.to(DEVICE)
        
        reconstruction = model(batch_frames) 
        mse = torch.mean((batch_frames - reconstruction) ** 2, dim=(1, 2, 3))
        reconstruction_errors.extend(mse.cpu().numpy())
        all_labels.extend(batch_labels.numpy())

# Convert to numpy arrays
reconstruction_errors = np.array(reconstruction_errors)
all_labels = np.array(all_labels)

# Compute regularity scores
regularity_scores = 1.0 / (1.0 + reconstruction_errors)

print(f"\nEvaluation complete!")
print(f"  Processed: {len(all_labels)} stacks")
print(f"  Normal (0): {np.sum(all_labels == 0)}")
print(f"  Anomalous (1): {np.sum(all_labels == 1)}\n")

# ----------------------------- CALCULATE METRICS -----------------------------
print("="*60 + "\n Metrics \n" + "="*60)


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
print("Regularity Score:")
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
print("="*60 + "\n Performance Metrics\n" + "="*60)

if len(np.unique(all_labels)) >= 2:
    auc_roc = roc_auc_score(all_labels, reconstruction_errors)
    print(f"  AUC-ROC Score: {auc_roc:.4f}")
    
else:
    auc_roc = None
    print("  AUC-ROC: Cannot compute (need both classes)")
    print("  STAUCC: Cannot compute (need both classes)")

print("="*60 + "\n")

# ----------------------------- Create out dir -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------- plot roc curve -----------------------------
if len(np.unique(all_labels)) >= 2:
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

# ----------------------------- PLOT SCORE DISTRIBUTIONS -----------------------------
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
print(f"Score distribution saved: {dist_path}")

# ==================== GENERATE TEXT REPORT ====================

report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')

with open(report_path, 'w') as f:
    f.write("="*60 + "\n ConvAE - Evaluation Report 10 stacks\n" + "="*60)
   
    
    f.write("Model Details\n")
    f.write("-"*60 + "\n")
    f.write(f"Model Path: {MODEL_PATH}\n")
    f.write(f"Image Stack Size: {config.IMAGE_STACK}\n")
    f.write(f"Image Size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}\n")
    f.write(f"Grayscale Channels: {config.CONVERT_GRAY}\n")
    f.write(f"Input Channels (stacks): {config.CONVERT_GRAY * config.IMAGE_STACK}\n\n")
    
    f.write("Dataset Statistics\n")
    f.write("-"*60 + "\n")
    f.write(f"Total stacks: {len(all_labels)}\n")
    f.write(f"Normal stacks (0): {np.sum(all_labels == 0)}\n")
    f.write(f"Anomalous stacks (1): {np.sum(all_labels == 1)}\n\n")
    
    f.write("Reconstruction Error Stats\n")
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
    else:
        f.write("AUC-ROC: Cannot compute (need both classes)\n")
    
    f.write("\n" + "="*60 + "\n")

print(f"Report Saved: {report_path}")

# ==================== SUMMARY ====================
print("="*60 + "\n Evaluation Complete \n" + "="*60)

print(f"Results saved to: {OUTPUT_DIR}/")
if auc_roc is not None:
    print(f"\nKey Metrics:")
    print(f"  AUC-ROC: {auc_roc:.4f}")

