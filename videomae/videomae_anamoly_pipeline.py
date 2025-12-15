import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import VideoMAEModel, VideoMAEImageProcessor
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import json
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt


import copy


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# ------------------------------------------- Model definition ------------------------------------------- #
class VideoMAEAnomalyDetector(nn.Module):
    def __init__(
        self, 
        pretrained_model: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        num_classes: int = 1,  # Normal vs Anomaly
        dropout: float = 0.3,
        freeze_layers  = 10  # 2 out of 12
    ):
        super(VideoMAEAnomalyDetector, self).__init__()
        
        self.videomae = VideoMAEModel.from_pretrained(pretrained_model)
        self._freeze_layers(freeze_layers)
        hidden_size = self.videomae.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    def _freeze_layers(self, freeze_layers):
        if freeze_layers > 0:
            # Always freeze embeddings
            for param in self.videomae.embeddings.parameters():
                param.requires_grad = False
            
            # Freeze first N encoder layers
            num_layers = len(self.videomae.encoder.layer)
            for i in range(min(freeze_layers, num_layers)):
                for param in self.videomae.encoder.layer[i].parameters():
                    param.requires_grad = False


    def forward(self, pixel_values):
        outputs = self.videomae(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :] 
        logits = self.classifier(cls_output)
        
        return logits

# ---------------------------------------------- DATA SET CLASSES ------------------------------------------ #
# Reading images on the fly  (SLOW)
class DoTAVideoMAEDataset(Dataset):
    def __init__(
        self, 
        root_dir,
        file_list,
        processor: VideoMAEImageProcessor = None,
        sequence_size=16, 
        overlap=0,         
        only_normal=True,  
        include_labels=True, 
    ):
        assert 0 <= overlap < sequence_size, "Overlap must be smaller than sequence size."
        self.root_dir = root_dir
        self.processor = processor
        self.only_normal = only_normal
        self.sequence_size = sequence_size
        self.include_labels = include_labels
        self.sequences = [] 

        self._sliding_skip = sequence_size - overlap
        
        # process all the jsonfiles
        annotations_path = os.path.join(root_dir, "annotations")
        for file in file_list:
            json_path = os.path.join(annotations_path, file)
            if not os.path.exists(json_path):
                print(f"Warning: JSON file doesn't exist: {json_path}")
                continue
            self._process_json_file(json_path)
        
        print(f"Total sequences created: {len(self.sequences)}")

        self._print_statistics()

    def _process_json_file(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        frames_data = []
        for label_info in data["labels"]:
            accident_name = label_info.get("accident_name", "normal")
            image_path = label_info.get("image_path", "")
            if not image_path:
                continue

            full_path = os.path.join(self.root_dir, image_path)
            label = 0 if accident_name == "normal" else 1

            frames_data.append({
                'path': full_path,
                'label': label,
                'accident_name': accident_name  # use this later if I wanna extra classifcation
            })
        
        # Now we generate sequences
        for i in range(0, len(frames_data), self._sliding_skip):
            sequence_frames = frames_data[i:i + self.sequence_size]
            
            if len(sequence_frames) < self.sequence_size:
                continue
            
            frame_paths = [f['path'] for f in sequence_frames]
            sequence_labels = [f['label'] for f in sequence_frames]
            sequence_label = 1 if any(sequence_labels) else 0
            self.sequences.append({
                'paths': frame_paths,
                'label': sequence_label
            })
    
    def _print_statistics(self):
        if not self.sequences:
            print("No sequences found!")
            return
        
        labels = [seq['label'] for seq in self.sequences]
        normal_count = sum(1 for l in labels if l == 0)
        anomaly_count = sum(1 for l in labels if l == 1)
        
        print(f"\nDataset Statistics:")
        print(f"  Total sequences: {len(self.sequences)}")
        print(f"  Normal sequences: {normal_count} ({normal_count/len(labels)*100:.1f}%)")
        print(f"  Anomaly sequences: {anomaly_count} ({anomaly_count/len(labels)*100:.1f}%)")

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        frame_paths = sequence['paths']
        label = sequence['label']

        frames = []
        for img_path in frame_paths:
            img = Image.open(img_path).convert("RGB")
            frames.append(np.array(img))
        
        if self.processor:
            inputs = self.processor(frames, return_tensors="pt")
    
            pixel_values = inputs['pixel_values'].squeeze(0)
            
        return pixel_values, torch.tensor(label, dtype=torch.float32)
  

# Using Preprocessed tensors for fast access (FAST)
class DoTAVideoMAEDatasetPreprocessed(Dataset):
    def __init__(self, preprocessed_dir):  
        self.preprocessed_dir = preprocessed_dir
        
        # Load metadata
        metadata_path = os.path.join(preprocessed_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded preprocessed dataset from: {preprocessed_dir}")
        print(f"  Total clips: {len(self.metadata)}")
        
        # Print statistics
        labels = [item['label'] for item in self.metadata]
        normal = sum(1 for l in labels if l == 0)
        anomaly = sum(1 for l in labels if l == 1)
        print(f"  Normal: {normal} ({normal/len(labels)*100:.1f}%)")
        print(f"  Anomaly: {anomaly} ({anomaly/len(labels)*100:.1f}%)")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        clip_file = item['clip_file']
        
        # Load preprocessed tensor
        clip_path = os.path.join(self.preprocessed_dir, clip_file)
        data = torch.load(clip_path)
        
        pixel_values = data['pixel_values']  # [16, 3, 224, 224]
        label = data['label']
        
        return pixel_values, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------- Helper Functionss ------------------------------------------ # 
def train_epoch(model, device, loader, criterion, optimizer):
    model.train()
    train_loss = 0
    for x, y in tqdm(loader, desc="Train Loader"):
        x = x.to(device)
        y = y.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(loader)


def valid_epoch(model, device, loader, criterion):
    model.eval()
    valid_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Valid Loader"):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            out = model(x)
            loss = criterion(out, y)
            valid_loss += loss.item()
            
            # Sigmoid and threshold
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
    
    avg_loss = valid_loss / len(loader)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # AUC (use probabilities, not predictions)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return avg_loss, accuracy, precision, recall, f1, auc


class Config:
    SEED = 42
    
    # Dataset Directories
    ROOT = "/home/public/mkamal/datasets/deep_learning/projdata/uploaded_data"
    JSON_FILE_PATH = "/home/grad/masters/2025/mkamal/mkamal/dl_project/anamoly-detection/train_val.txt"
    PREPROCESSED_TRAIN="/home/public/mkamal/datasets/deep_learning/projdata/uploaded_data/preprocessed_clips/train"
    PREPROCESSED_VALID="/home/public/mkamal/datasets/deep_learning/projdata/uploaded_data/preprocessed_clips/validation"

    # Dataset
    BATCH = 64
    NUM_WORKERS = 24
    CLIP_SIZE = 16          
    SLIDE_OVERLAP = 0

    # TRAINING SETUP
    EPOCH = 6
    LR = 6.20e-04  # Learning Rate prev: 1e-4  , 6.20e-04
    DECAY = 1e-2   # Regularization Parameter
    EARLY_STOPPING_PATIENCE = 2   # 3 epoch patience

    # Model Setup
    PRETRAINED_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics" 
    NUM_CLASSES = 1
    FREEZE_LAYERS = 9  # prev: 10 of 12
    FREEZE_LAYERS = min(FREEZE_LAYERS, 12) # can maximally freeze 12 layers
    DROPOUT = 0.5

    # Model Output Path (Modify these)
    SAVE_PATH = "/home/public/mkamal/saved_models/" + f"videomae_b{BATCH}_e{EPOCH}.pth"
    PLOT_SAVE_PATH = "/home/grad/masters/2025/mkamal/mkamal/dl_project/anamoly-detection/ConvAE/" + f"videomae_b{BATCH}_e{EPOCH}_train_valild_plot.png"
    



def main():
    config = Config()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device: ", device)

    # ------------------------------------------ DATA SETUP ------------------------------------------ # 
    print("="*60 + "\n Setting up Dataset \n"+ "="*60)

    processor = VideoMAEImageProcessor.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics"
    )

    train_ds = DoTAVideoMAEDatasetPreprocessed(config.PREPROCESSED_TRAIN)
    valid_ds = DoTAVideoMAEDatasetPreprocessed(config.PREPROCESSED_VALID)

    train_loader = DataLoader(train_ds, 
                              batch_size=config.BATCH, 
                              shuffle=True, 
                              num_workers=config.NUM_WORKERS, 
                              pin_memory=True,
                              )
    val_loader   = DataLoader(valid_ds, 
                              batch_size=config.BATCH, 
                              shuffle=False, 
                              num_workers=config.NUM_WORKERS, 
                              pin_memory=True)


    # ------------------------------------------ MODEL SETUP ------------------------------------------ #
    print("="*60 + "\n Setting up Model \n"+ "="*60)
    model  = VideoMAEAnomalyDetector(
        pretrained_model=config.PRETRAINED_MODEL, 
        num_classes=config.NUM_CLASSES, 
        freeze_layers=config.FREEZE_LAYERS,
        dropout=config.DROPOUT).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.DECAY)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Criterion Used: {criterion.__class__.__name__}")
    print(f"Optimizer Used: {optimizer}")
    # ------------------------------------------  TRAIN LOOP ------------------------------------------ # 
    print( "="*60 + "\n Starting Training \n" + "="*60)
    patience=config.EARLY_STOPPING_PATIENCE; best=float("inf"); waited=0; best_state=None

    tl_list=[]; vl_list=[]; vl_acc = []
    for epoch in range(config.EPOCH):
        train_loss = train_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _, _, val_auc = valid_epoch(
            model, device, val_loader, criterion
        )
        tl_list.append(train_loss)
        vl_list.append(val_loss)
        vl_acc.append(vl_acc)

        # Early Stopping logic
        if val_loss < best - 1e-5: 
            best=val_loss
            waited=0
            best_state=copy.deepcopy(model.state_dict())
        else:
            waited+=1
            if waited>patience:
                print("Early stopping.")
                break
        
        scheduler.step(val_loss)        
        print(f"Epoch {epoch+1}/{config.EPOCH} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

    

   
    # ------------------------------------------  TRAIN LOOP ------------------------------------------ # 
    print("="*60 + "\n Saving & Plotting \n"+ "="*60)

    save_path = config.SAVE_PATH
    torch.save(best_state, save_path)
    print(f"Model saved in path: {save_path}")

    plt.figure()
    plt.plot(tl_list, label="Train Loss")
    plt.plot(vl_list, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Metrics: videomae + classifier")
    plt.grid(True, alpha=0.3)
    plt.savefig(config.PLOT_SAVE_PATH)
    plt.close()  # Close first figure


    # Validation accuracy plot
    plt.figure()
    plt.plot(vl_acc, label="Validation Accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy: videomae + classifier")
    plt.grid(True, alpha=0.3)
    plt.savefig("videomae_val_acc.png")
    plt.close()


if __name__ == "__main__":
    main() 
    