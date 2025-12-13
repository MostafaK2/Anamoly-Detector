import os
from sklearn.model_selection import train_test_split

# goes into annotations folder and split the test json file paths and train json file path
def split_by_json_file(
        root="/home/public/mkamal/datasets/deep_learning/projdata/uploaded_data", 
        random_state = 42): 
    
    annotations_path = os.path.join(root, "annotations")
    json_files = sorted([f for f in os.listdir(annotations_path) if f.endswith(".json")])
    train_val_files, test_files = train_test_split(json_files, test_size=0.1, random_state=random_state)

    # Save train+val list in current directory
    with open("train_val.txt", "w") as f:
        for item in train_val_files:
            f.write("%s\n" % item)

    # Save test list in current directory
    with open("test.txt", "w") as f:
        for item in test_files:
            f.write("%s\n" % item)
        
    print("Sucessfully outputted train-valid and testing files into disk")

    return train_val_files, test_files


def read_json_files(json_file_path):
    if (not os.path.exists(json_file_path)):
        raise FileNotFoundError(f"No JSON file found at {json_file_path}")
    with open(json_file_path, "r") as f:
        json_files = [line.strip() for line in f.readlines()]
    
    return json_files


# RUN THIS ONCE --- DONE
def scan_and_export_json_files(
        annotations_folder_dir = "/home/public/mkamal/datasets/deep_learning/projdata/uploaded_data/annotations", 
        output_file="video_json_files.txt"):
    json_files = []

    if not os.path.exists(annotations_folder_dir):
        raise FileNotFoundError(f"Directory not found: {annotations_folder_dir}")
    if not os.path.isdir(annotations_folder_dir): 
        raise NotADirectoryError(f"Path is not a directory: {annotations_folder_dir}")
    
    for _, _, files in os.walk(annotations_folder_dir):
        for file_name in files:
            if not file_name.endswith(".json"):
                continue
            
            json_files.append(file_name)
    
    # Write to output file
    with open(output_file, 'w') as f:
        for json_file in json_files:
            f.write(json_file + '\n')
    
    print(f"Found {len(json_files)} JSON files. Written to {output_file}")
    return json_files

def split_before_training(json_file, split_ratio, random_state=42):
    data = read_json_files(json_file)
    train_files, test_files = train_test_split(data, test_size=split_ratio, random_state=random_state)

    return train_files, test_files





















import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy


def lr_range_test_autoencoder(model, train_loader, device, start_lr=1e-7, end_lr=1, 
                               num_iter=1000, smooth_factor=0.05, divergence_threshold=4):
    """
    Learning Rate Range Test (LR Finder) for Autoencoder
    
    Systematically increases learning rate and tracks reconstruction loss to find optimal LR.
    
    Args:
        model: Your autoencoder model
        train_loader: Training data loader
        device: cuda or cpu
        start_lr: Starting learning rate (default: 1e-7)
        end_lr: Ending learning rate (default: 1)
        num_iter: Number of iterations (default: one epoch)
        smooth_factor: Smoothing factor for loss curve (default: 0.05)
        divergence_threshold: Stop if loss exceeds best_loss * threshold
    
    Returns:
        lrs: List of learning rates tested
        losses: List of corresponding losses
        suggested_lr: Suggested starting learning rate
    """
    print("\n" + "="*100)
    print("LEARNING RATE RANGE TEST - ConvAE")
    print("="*100)
    print(f"Range: {start_lr:.2e} → {end_lr:.2e}")
    
    # Make a copy of the model to avoid messing up the original
    model_copy = copy.deepcopy(model)
    model_copy.train()
    
    # Setup - MSE for reconstruction
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_copy.parameters(), lr=start_lr)
    
    # Calculate number of iterations
    if num_iter is None:
        num_iter = len(train_loader)
    
    # Calculate LR multiplication factor
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    
    # Storage
    lrs = []
    losses = []
    best_loss = float('inf')
    avg_loss = 0
    batch_num = 0
    
    # Iterate through training data
    iterator = iter(train_loader)
    
    print(f"\nTesting {num_iter} iterations...")
    
    for iteration in tqdm(range(num_iter), desc="LR Range Test"):
        batch_num += 1
        
        # Get batch
        try:
            batch = next(iterator)
        except StopIteration:
            # If we run out of data, restart iterator
            iterator = iter(train_loader)
            batch = next(iterator)
        
        inp = batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        out = model_copy(inp)
        loss = criterion(out, inp)
        
        # Compute smoothed loss
        if iteration == 0:
            avg_loss = loss.item()
        else:
            avg_loss = smooth_factor * loss.item() + (1 - smooth_factor) * avg_loss
        
        # Track best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # Stop if loss is diverging
        if avg_loss > divergence_threshold * best_loss:
            print(f"\n⚠️  Stopping early - loss is diverging (iteration {iteration})")
            break
        
        # Record
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(avg_loss)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult
    
    print("\n✅ LR Range Test Complete!")
    
    # Find suggested learning rate
    suggested_lr = suggest_lr(lrs, losses)
    
    # Plot results
    plot_lr_finder(lrs, losses, suggested_lr)
    
    return lrs, losses, suggested_lr


def suggest_lr(lrs, losses):
    """
    Suggest optimal learning rate based on the steepest descent
    """
    # Find the steepest negative gradient
    gradients = np.gradient(losses)
    min_gradient_idx = np.argmin(gradients)
    
    # Suggested LR is typically where gradient is steepest (before minimum)
    # Use LR that's ~10x smaller than where loss is minimum
    min_loss_idx = np.argmin(losses)
    
    # Take the LR at steepest descent, or 1/10th of min loss LR
    if min_gradient_idx < len(lrs) * 0.8:  # Only if not too close to end
        suggested_idx = min_gradient_idx
    else:
        suggested_idx = max(0, min_loss_idx - len(lrs) // 10)
    
    suggested_lr = lrs[suggested_idx]
    
    return suggested_lr


def plot_lr_finder(lrs, losses, suggested_lr):
    """
    Plot the learning rate range test results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Loss vs Learning Rate (log scale)
    ax1.plot(lrs, losses, linewidth=2, color='#2E86AB')
    ax1.set_xscale('log')
    ax1.set_xlabel('Learning Rate (log scale)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MSE Loss (smoothed)', fontsize=12, fontweight='bold')
    ax1.set_title('Learning Rate Range Test - ConvAE', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Mark suggested LR
    ax1.axvline(x=suggested_lr, color='red', linestyle='--', linewidth=2, 
                label=f'Suggested LR: {suggested_lr:.2e}')
    ax1.legend(fontsize=11)
    
    # Plot 2: Loss vs Iteration
    ax2.plot(losses, linewidth=2, color='#A23B72')
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MSE Loss (smoothed)', fontsize=12, fontweight='bold')
    ax2.set_title('Loss Over Iterations', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convAE_lr_range_test.png', dpi=300, bbox_inches='tight')
    print("\n📊 Plot saved as 'convAE_lr_range_test.png'")
    plt.show()
    
    # Print recommendations
    print("\n" + "="*100)
    print("LEARNING RATE RECOMMENDATIONS")
    print("="*100)
    print(f"\n📍 Suggested Starting LR: {suggested_lr:.2e}")
    print(f"\n💡 Guidelines:")
    print(f"   • Start training with LR around: {suggested_lr:.2e}")
    print(f"   • Consider trying: {suggested_lr/3:.2e} (more conservative)")
    print(f"   • Or try: {suggested_lr*3:.2e} (more aggressive)")
    print(f"\n⚠️  Look for:")
    print(f"   • LR where loss decreases fastest (steepest downward slope)")
    print(f"   • Pick a value BEFORE the loss starts increasing")
    print(f"   • Common choice: 1/10th of the LR at minimum loss")
    print("="*100 + "\n")






