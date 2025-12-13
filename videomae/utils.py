import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy


def lr_range_test_videomae(model, train_loader, device, start_lr=1e-7, end_lr=1, num_iter=300, smooth_factor=0.05, divergence_threshold=4):
    print(f"Range: {start_lr:.2e} → {end_lr:.2e}")
 
    model_copy = copy.deepcopy(model);model_copy.train()
    criterion = nn.BCEWithLogitsLoss();optimizer = optim.Adam(model_copy.parameters(), lr=start_lr)
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)  # Learning rate multiplyer
    
    lrs, losses = [],[]
    best_loss = float('inf')
    avg_loss = 0
    batch_num = 0
    
    # Iterate through training data
    iterator = iter(train_loader)
    print(f"\nTesting {num_iter} iterations")
    
    for iteration in tqdm(range(num_iter), desc="LR Range Test"):
        batch_num += 1
        try:
            x, y = next(iterator)
        except StopIteration:
            # If we run out of data, restart iterator
            iterator = iter(train_loader)
            x, y = next(iterator)
        
        x = x.to(device)
        y = y.to(device).unsqueeze(1) 
        optimizer.zero_grad()
        logits = model_copy(x)
        loss = criterion(logits, y)
        
        # Compute smoothed loss
        if iteration == 0:
            avg_loss = loss.item()
        else:
            avg_loss = smooth_factor * loss.item() + (1 - smooth_factor) * avg_loss
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # Stop if loss is diverging
        if avg_loss > divergence_threshold * best_loss:
            print(f"\nStopping early - loss is diverging (iteration {iteration})")
            break
        
        # Record
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(avg_loss)

        loss.backward()
        optimizer.step()
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult
    
    print("\n LR Range Test Complete!")
    
    suggested_lr = suggest_lr(lrs, losses)
    plot_lr_finder(lrs, losses, suggested_lr, model_name="VideoMAE")
    
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


def plot_lr_finder(lrs, losses, suggested_lr, model_name="VideoMAE"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Loss vs Learning Rate (log scale)
    ax1.plot(lrs, losses, linewidth=2, color='#2E86AB')
    ax1.set_xscale('log')
    ax1.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('BCE Loss', fontsize=12, fontweight='bold')
    ax1.set_title(f'Learning Rate Range Test: {model_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Mark suggested LR
    ax1.axvline(x=suggested_lr, color='red', linestyle='--', linewidth=2, 
                label=f'Suggested LR: {suggested_lr:.2e}')
    ax1.legend(fontsize=11)
    
    # Plot 2: Loss vs Iteration
    ax2.plot(losses, linewidth=2, color='#A23B72')
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('BCE Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Loss Over Iterations', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('videomae_lr_range_test.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'videomae_lr_range_test.png'")
    plt.close()
    

# Add this to your main training file
def run_lr_finder():
    """
    Run LR finder before training
    """
    from trainVideoMAEAnamoly import Config, DoTAVideoMAEDatasetPreprocessed, VideoMAEAnomalyDetector
    from torch.utils.data import DataLoader
    
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Setting up LR Finder...")
    
    # Load dataset
    train_ds = DoTAVideoMAEDatasetPreprocessed(config.PREPROCESSED_TRAIN)
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.BATCH,  # Use same batch size as training
        shuffle=True, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=True
    )
    
    model = VideoMAEAnomalyDetector(
        pretrained_model=config.PRETRAINED_MODEL,
        num_classes=config.NUM_CLASSES,
        freeze_backbone=config.FREEZE_BACKBONE,
        freeze_layers=config.FREEZE_LAYERS
    ).to(device)
    
    # Run LR finder
    lrs, losses, suggested_lr = lr_range_test_videomae(
        model=model,
        train_loader=train_loader,
        device=device,
        start_lr=1e-8,
        end_lr=3e-3,   # between 1e-2 and 1e-3
        num_iter=120,
        smooth_factor=0.05,
        divergence_threshold=4
    )
    
    return suggested_lr


if __name__ == "__main__":
    suggested_lr = run_lr_finder()
    
    print(f"\nUse this LR in your config: {suggested_lr:.2e}")