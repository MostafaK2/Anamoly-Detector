import torch
import numpy as np
from transformers import VideoMAEImageProcessor
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from torch.utils.data import DataLoader
from videomae_anamoly_pipeline import Config, DoTAVideoMAEDatasetPreprocessed, VideoMAEAnomalyDetector, DoTAVideoMAEDataset
from tqdm import tqdm

def read_json_files(json_file_path):
    if (not os.path.exists(json_file_path)):
        raise FileNotFoundError(f"No JSON file found at {json_file_path}")
    with open(json_file_path, "r") as f:
        json_files = [line.strip() for line in f.readlines()]
    
    return json_files


def evaluate_videomae_model(model, test_loader, device, threshold=0.5):
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    print("="*60,"\n Running inference on test set \n", "="*60)
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Move to device
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs >= threshold).astype(int)
            
            # Store results
            all_probabilities.extend(probs)
            all_predictions.extend(preds)
            all_labels.extend(y.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    print("\nCalculating metrics...")
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    auc_roc = roc_auc_score(all_labels, all_probabilities)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Per-class metrics
    class_report = classification_report(
        all_labels, all_predictions, 
        target_names=['Normal', 'Anomaly'],
        output_dict=True,
        zero_division=0
    )
    
    # Compile results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': cm,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'class_report': class_report,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'labels': all_labels
    }
    
    return results

    """
    Print evaluation results in a formatted way
    """
    print("\n" + "="*60)
    print("VideoMAE Model Evaluation Results")
    print("="*60)
    
    print("\n--- Overall Metrics ---")
    print(f"Accuracy:   {results['accuracy']:.4f}")
    print(f"Precision:  {results['precision']:.4f}")
    print(f"Recall:     {results['recall']:.4f}")
    print(f"F1-Score:   {results['f1_score']:.4f}")
    print(f"AUC-ROC:    {results['auc_roc']:.4f}")
    
    print("\n--- Confusion Matrix ---")
    print(f"                Predicted")
    print(f"                Normal  Anomaly")
    print(f"Actual Normal   {results['tn']:6d}  {results['fp']:6d}")
    print(f"       Anomaly  {results['fn']:6d}  {results['tp']:6d}")
    
    print("\n--- Per-Class Metrics ---")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print("-" * 65)
    
    for class_name in ['Normal', 'Anomaly']:
        metrics = results['class_report'][class_name]
        print(f"{class_name:<15} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1-score']:<12.4f} "
              f"{int(metrics['support']):<12d}")
    
    # Weighted average
    weighted = results['class_report']['weighted avg']
    print("-" * 65)
    print(f"{'Weighted Avg':<15} {weighted['precision']:<12.4f} "
          f"{weighted['recall']:<12.4f} {weighted['f1-score']:<12.4f} "
          f"{int(weighted['support']):<12d}")
    
    print("\n" + "="*60)

def save_results_to_file(results, filename='/videomae/videomae_results.txt'):
    with open(filename, 'w') as f:
        f.write("VideoMAE Model Evaluation Results\n")
        f.write("="*60 + "\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"Accuracy:   {results['accuracy']:.4f}\n")
        f.write(f"Precision:  {results['precision']:.4f}\n")
        f.write(f"Recall:     {results['recall']:.4f}\n")
        f.write(f"F1-Score:   {results['f1_score']:.4f}\n")
        f.write(f"AUC-ROC:    {results['auc_roc']:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(f"TN: {results['tn']}, FP: {results['fp']}\n")
        f.write(f"FN: {results['fn']}, TP: {results['tp']}\n\n")
        
        f.write("Per-Class Metrics:\n")
        for class_name in ['Normal', 'Anomaly']:
            metrics = results['class_report'][class_name]
            f.write(f"{class_name}: Precision={metrics['precision']:.4f}, "
                   f"Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}, "
                   f"Support={int(metrics['support'])}\n")
    
    print(f"\nResults saved to {filename}")

# Main execution
if __name__ == "__main__":
    # Configuration
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test dataset (you'll need to create a preprocessed test directory)
    print("="*60, "\nLoading test dataset\n", "="*60)
    TEST = "/home/grad/masters/2025/mkamal/mkamal/dl_project/anamoly-detection/test.txt"
    
    processor = VideoMAEImageProcessor.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics"
    )

    test_files = read_json_files(TEST)
    test_dataset = DoTAVideoMAEDataset(root_dir=config.ROOT, file_list=test_files,processor=processor)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Load trained model
    print("="*60,"\nLoading trained model\n", "="*60)
    model = VideoMAEAnomalyDetector(
        pretrained_model=config.PRETRAINED_MODEL,
        num_classes=config.NUM_CLASSES,
        freeze_layers=config.FREEZE_LAYERS,
        dropout=config.DROPOUT
    )
    
    # Load best checkpoint
    checkpoint_path = '/home/public/mkamal/saved_models/videomaebest.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Using device: {device}")
    
    # Evaluate model
    results = evaluate_videomae_model(
        model=model,
        test_loader=test_loader,
        device=device,
        threshold=0.5
    )
        
    # Save results
    save_results_to_file(results, '/home/grad/masters/2025/mkamal/mkamal/dl_project/anamoly-detection/videomae/videomae_evaluation_results.txt')
    