import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import VideoMAEImageProcessor
import os
from videomae_anamoly_pipeline import Config, DoTAVideoMAEDataset, VideoMAEAnomalyDetector
from tqdm import tqdm
from PIL import Image

def read_json_files(json_file_path):
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"No JSON file found at {json_file_path}")
    with open(json_file_path, "r") as f:
        json_files = [line.strip() for line in f.readlines()]
    return json_files

def get_temporal_predictions_with_frames(model, dataset, device, num_samples=5):
    model.eval()
    model.to(device)
    
    temporal_data = []
    
    with torch.no_grad():
        for idx in tqdm(range(min(num_samples * 20, len(dataset))), desc="Getting predictions"):
            x, y = dataset[idx]
            x_batch = x.unsqueeze(0).to(device)
            
            output = model(x_batch)
            prob = torch.sigmoid(output).cpu().numpy()[0][0]
            
            # Get the sequence info to access frame paths
            sequence = dataset.sequences[idx]
            frame_paths = sequence['paths']
            
            temporal_data.append({
                'idx': idx,
                'probability': prob,
                'label': y.item(),
                'frame_paths': frame_paths,  # Store frame paths
                'middle_frame': frame_paths[len(frame_paths)//2]  # Middle frame of sequence
            })
    
    return temporal_data

### PLOT GENERATED WITH THE HELP OF LLM ### 
def plot_temporal_with_frames(temporal_data, window_size=100, save_dir='temporal_plots'):
    os.makedirs(save_dir, exist_ok=True)
    
    num_windows = len(temporal_data) // window_size
    
    for window_idx in range(min(num_windows, 5)):
        start_idx = window_idx * window_size
        end_idx = start_idx + window_size
        
        window_data = temporal_data[start_idx:end_idx]
        
        frame_indices = [d['idx'] for d in window_data]
        predictions = [d['probability'] for d in window_data]
        ground_truth = [d['label'] for d in window_data]
        
        # Create figure with temporal plot and sample frames
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        # Main temporal plot (top, spanning all columns)
        ax_main = fig.add_subplot(gs[0:2, :])
        
        # Plot anomaly score over time
        ax_main.plot(frame_indices, predictions, 'b-', linewidth=2, 
                     label='Anomaly Score', alpha=0.8)
        
        threshold = 0.5
        ax_main.axhline(y=threshold, color='r', linestyle='--', 
                        linewidth=1.5, label=f'Threshold ({threshold})')
        
        # Shade ground truth anomaly regions
        for i in range(len(ground_truth)):
            if ground_truth[i] == 1:
                ax_main.axvspan(frame_indices[i]-0.5, frame_indices[i]+0.5, 
                               alpha=0.2, color='red')
        
        from matplotlib.patches import Patch
        red_patch = Patch(color='red', alpha=0.2, label='Ground Truth Anomaly')
        handles, labels = ax_main.get_legend_handles_labels()
        handles.append(red_patch)
        
        ax_main.set_xlabel('Sequence Index', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('Anomaly Score', fontsize=12, fontweight='bold')
        ax_main.set_title(f'Temporal Anomaly Detection - Window {window_idx + 1}', 
                         fontsize=14, fontweight='bold')
        ax_main.set_ylim([-0.05, 1.05])
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(handles=handles, loc='upper right', fontsize=10)
        
        # Sample frames (bottom, 2 rows x 4 columns = 8 frames)
        sample_indices = np.linspace(0, len(window_data)-1, 8, dtype=int)
        
        for i, sample_idx in enumerate(sample_indices):
            row = 2 + i // 4
            col = i % 4
            ax_frame = fig.add_subplot(gs[row, col])
            
            # Load and display middle frame from sequence
            frame_path = window_data[sample_idx]['middle_frame']
            try:
                img = Image.open(frame_path).convert('RGB')
                ax_frame.imshow(img)
                ax_frame.axis('off')
                
                # Add info text
                seq_idx = window_data[sample_idx]['idx']
                prob = window_data[sample_idx]['probability']
                label = 'Anomaly' if window_data[sample_idx]['label'] == 1 else 'Normal'
                pred = 'Anomaly' if prob >= threshold else 'Normal'
                
                color = 'green' if (pred == label) else 'red'
                ax_frame.set_title(f"Seq {seq_idx}\nGT: {label} | Pred: {pred}\nScore: {prob:.2f}", 
                                  fontsize=8, color=color, fontweight='bold')
                
                # Mark position on main plot
                ax_main.axvline(x=seq_idx, color='purple', alpha=0.3, linestyle=':')
                
            except Exception as e:
                ax_frame.text(0.5, 0.5, 'Image\nNot Found', 
                             ha='center', va='center', fontsize=10)
                ax_frame.axis('off')
        
        plt.savefig(f'{save_dir}/temporal_window_{window_idx + 1}_with_frames.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved temporal plot with frames {window_idx + 1}")

### PLOT GENERATED WITH THE HELP OF LLM ### 
def plot_scenarios_with_frames(temporal_data, save_dir='temporal_plots'):
    os.makedirs(save_dir, exist_ok=True)
    
    threshold = 0.5
    
    # Find examples
    true_positives = []
    false_positives = []
    false_negatives = []
    true_negatives = []
    
    for i, data in enumerate(temporal_data):
        pred = 1 if data['probability'] >= threshold else 0
        label = data['label']
        
        if pred == 1 and label == 1:
            true_positives.append(i)
        elif pred == 1 and label == 0:
            false_positives.append(i)
        elif pred == 0 and label == 1:
            false_negatives.append(i)
        else:
            true_negatives.append(i)
    
    scenarios = [
        ('True Positive', true_positives[:2]),
        ('False Positive', false_positives[:2]),
        ('False Negative', false_negatives[:2]),
        ('True Negative', true_negatives[:2])
    ]
    
    for scenario_name, indices in scenarios:
        if not indices:
            continue
        
        for example_num, center_idx in enumerate(indices):
            # Create figure with temporal context and frame sequence
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
            
            # Temporal context plot (top)
            ax_temporal = fig.add_subplot(gs[0, :])
            
            window = 50
            start = max(0, center_idx - window)
            end = min(len(temporal_data), center_idx + window)
            
            window_data = temporal_data[start:end]
            frame_indices = [d['idx'] for d in window_data]
            predictions = [d['probability'] for d in window_data]
            ground_truth = [d['label'] for d in window_data]
            
            ax_temporal.plot(frame_indices, predictions, 'b-', linewidth=2, alpha=0.8)
            ax_temporal.axhline(y=threshold, color='r', linestyle='--', linewidth=1.5)
            
            for i in range(len(ground_truth)):
                if ground_truth[i] == 1:
                    ax_temporal.axvspan(frame_indices[i]-0.5, frame_indices[i]+0.5, 
                                       alpha=0.2, color='red')
            
            # Highlight the detection point
            center_in_window = center_idx - start
            if 0 <= center_in_window < len(frame_indices):
                ax_temporal.axvline(x=frame_indices[center_in_window], color='green', 
                                   linestyle='--', linewidth=2, label='Detection Point')
            
            ax_temporal.set_xlabel('Sequence Index', fontsize=11)
            ax_temporal.set_ylabel('Anomaly Score', fontsize=11)
            ax_temporal.set_ylim([-0.05, 1.05])
            ax_temporal.grid(True, alpha=0.3)
            ax_temporal.legend(loc='upper right')
            ax_temporal.set_title(f'{scenario_name} - Example {example_num + 1}', 
                                 fontsize=14, fontweight='bold')
            
            # Show 8 frames from the actual sequence
            sequence_frames = temporal_data[center_idx]['frame_paths']
            sample_frame_indices = np.linspace(0, len(sequence_frames)-1, 8, dtype=int)
            
            for i, frame_idx in enumerate(sample_frame_indices):
                row = 1 + i // 4
                col = i % 4
                ax_frame = fig.add_subplot(gs[row, col])
                
                try:
                    img = Image.open(sequence_frames[frame_idx]).convert('RGB')
                    ax_frame.imshow(img)
                    ax_frame.axis('off')
                    ax_frame.set_title(f'Frame {frame_idx+1}/{len(sequence_frames)}', 
                                      fontsize=9)
                except:
                    ax_frame.text(0.5, 0.5, 'Frame\nNot Found', 
                                 ha='center', va='center')
                    ax_frame.axis('off')
            
            # Add summary info
            prob = temporal_data[center_idx]['probability']
            label = 'Anomaly' if temporal_data[center_idx]['label'] == 1 else 'Normal'
            pred = 'Anomaly' if prob >= threshold else 'Normal'
            
            fig.text(0.5, 0.02, 
                    f"Ground Truth: {label} | Prediction: {pred} | Confidence: {prob:.3f}",
                    ha='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            filename = f"{scenario_name.lower().replace(' ', '_')}_example_{example_num + 1}.png"
            plt.savefig(f'{save_dir}/{filename}', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved {scenario_name} example {example_num + 1} with frames")

### PLOT GENERATED WITH THE HELP OF LLM ### 
def plot_statistics_over_time(temporal_data, save_dir='temporal_plots'):
    """
    Plot statistics with sample frames from different score ranges
    """
    os.makedirs(save_dir, exist_ok=True)
    
    window = 20
    frame_indices = [d['idx'] for d in temporal_data]
    predictions = [d['probability'] for d in temporal_data]
    labels = [d['label'] for d in temporal_data]
    
    rolling_mean = []
    for i in range(len(predictions)):
        start = max(0, i - window)
        rolling_mean.append(np.mean(predictions[start:i+1]))
    
    # Create figure with stats and sample frames
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
    
    # Plot 1: Raw scores vs rolling mean
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(frame_indices, predictions, 'b-', alpha=0.3, linewidth=1, label='Raw Scores')
    ax1.plot(frame_indices, rolling_mean, 'r-', linewidth=2, label=f'Rolling Mean (window={window})')
    ax1.axhline(y=0.5, color='k', linestyle='--', linewidth=1, label='Threshold')
    ax1.set_xlabel('Sequence Index', fontsize=12)
    ax1.set_ylabel('Anomaly Score', fontsize=12)
    ax1.set_title('Anomaly Scores with Rolling Average', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([-0.05, 1.05])
    
    # Plot 2: Distribution histogram
    ax2 = fig.add_subplot(gs[1, :])
    normal_scores = [predictions[i] for i in range(len(predictions)) if labels[i] == 0]
    anomaly_scores = [predictions[i] for i in range(len(predictions)) if labels[i] == 1]
    
    ax2.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
    ax2.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red', density=True)
    ax2.axvline(x=0.5, color='k', linestyle='--', linewidth=2, label='Threshold')
    ax2.set_xlabel('Anomaly Score', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Score Distribution by Class', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Sample frames from different score ranges
    # Low confidence normal (0.0-0.3), High confidence normal (0.7-1.0), 
    # Low confidence anomaly (0.3-0.5), High confidence anomaly (0.5-1.0)
    score_ranges = [
        (0.0, 0.3, "High Confidence Normal"),
        (0.7, 1.0, "High Confidence Anomaly"),
        (0.3, 0.5, "Uncertain (Near Threshold)"),
        (0.5, 0.7, "Moderate Confidence")
    ]
    
    for range_idx, (min_score, max_score, range_name) in enumerate(score_ranges):
        # Find examples in this range
        examples = [i for i, p in enumerate(predictions) 
                   if min_score <= p < max_score]
        
        if examples:
            sample_idx = examples[len(examples)//2]  # Middle example
            
            row = 2 + range_idx // 2
            col = (range_idx % 2) * 2
            
            # Show 2 frames from this sequence
            for frame_offset in range(2):
                ax_frame = fig.add_subplot(gs[row, col + frame_offset])
                
                frame_paths = temporal_data[sample_idx]['frame_paths']
                frame_idx = frame_offset * (len(frame_paths) // 2)
                
                try:
                    img = Image.open(frame_paths[frame_idx]).convert('RGB')
                    ax_frame.imshow(img)
                    ax_frame.axis('off')
                    
                    if frame_offset == 0:
                        prob = temporal_data[sample_idx]['probability']
                        label = 'Anomaly' if temporal_data[sample_idx]['label'] == 1 else 'Normal'
                        ax_frame.set_title(f"{range_name}\nGT: {label}, Score: {prob:.2f}", 
                                          fontsize=9, fontweight='bold')
                except:
                    ax_frame.axis('off')
    
    plt.savefig(f'{save_dir}/temporal_statistics_with_frames.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved temporal statistics plot with frames")

# Main execution
if __name__ == "__main__":
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60, "\nLoading test dataset\n", "="*60)
    TEST = "/home/grad/masters/2025/mkamal/mkamal/dl_project/anamoly-detection/test.txt"
    
    processor = VideoMAEImageProcessor.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics"
    )
    
    test_files = read_json_files(TEST)
    test_dataset = DoTAVideoMAEDataset(
        root_dir=config.ROOT, 
        file_list=test_files, 
        processor=processor,
        sequence_size=config.CLIP_SIZE,
        overlap=config.SLIDE_OVERLAP
    )
    print("="*60, "\nLoading Pretrained Model\n", "="*60)
    model = VideoMAEAnomalyDetector(
        pretrained_model=config.PRETRAINED_MODEL,
        num_classes=config.NUM_CLASSES,
        freeze_layers=config.FREEZE_LAYERS,
        dropout=config.DROPOUT
    )
    
    checkpoint_path = '/home/public/mkamal/saved_models/videomaebest.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    
    print(f"Model loaded from {checkpoint_path}")
    
    # Get temporal predictions with frame paths
    print("\nGenerating temporal predictions with frames")
    temporal_data = get_temporal_predictions_with_frames(model, test_dataset, device, num_samples=10)
    
    save_dir = "/home/grad/masters/2025/mkamal/mkamal/dl_project/anamoly-detection/videomae/"
    # Create visualizations with frames
    plot_temporal_with_frames(temporal_data, window_size=100, save_dir=save_dir+'temporal_plots')
    plot_scenarios_with_frames(temporal_data, save_dir=save_dir+'temporal_plots')
    plot_statistics_over_time(temporal_data, save_dir=save_dir+'temporal_plots')
    
