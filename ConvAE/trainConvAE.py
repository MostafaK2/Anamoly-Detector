import os
import json
import copy
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms
from torchvision.io import read_image

from tqdm import tqdm

from sklearn.model_selection import train_test_split

# ---------------------------------------- Utils Functions ---------------------------------------- #
def split_before_training(json_file, split_ratio, random_state=42):
    data = read_json_files(json_file)
    train_files, test_files = train_test_split(data, test_size=split_ratio, random_state=random_state)

    return train_files, test_files


def read_json_files(json_file_path):
    if (not os.path.exists(json_file_path)):
        raise FileNotFoundError(f"No JSON file found at {json_file_path}")
    with open(json_file_path, "r") as f:
        json_files = [line.strip() for line in f.readlines()]
    
    return json_files



# ---------------------------------------- MODEL CLASS ------------------------------------------- #
class Conv2DAutoEncoder(nn.Module):
	def __init__(self, input_shape):
		super(Conv2DAutoEncoder, self).__init__()
		
		self.encoder = nn.Sequential(
			# Conv1 -> BN -> RELU  -> MaxPool
			nn.Conv2d(input_shape, 512, 11, stride=4),   
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2,stride=2),    				 
			
            # Conv2 -> BN -> RELU -> Max Pool
			nn.Conv2d(512,256,5,stride =1,padding=2),    
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2,stride =2),                 
			
            # Conv3 -> BN -> RELU
            nn.Conv2d(256,128,3,stride =1,padding=1),  
		    nn.BatchNorm2d(128),
			nn.ReLU(inplace=True)
			
        )
		self.decoder = nn.Sequential(
			# Deconv Layer1
			nn.ConvTranspose2d(128,128,3,stride=1,padding=1),  
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(128,128,2,stride=2,dilation=2), 
			
            nn.ConvTranspose2d(128,256,3,stride=1,padding=1),   
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(256,256,2,stride=2, dilation=2),  
			
            nn.ConvTranspose2d(256,512,5,stride=1, padding=2),  
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(512, input_shape, 11, stride=4, padding=2, output_padding=1)  

        )
	def forward(self,img):
		img = self.encoder(img)
		img = self.decoder(img)
		return img.contiguous()


# ---------------------------------------- DATASET CLASS ----------------------------------------- #
# Very slow but works
class StackedFramesDataset(Dataset):
    """
    Dataset that returns 10 consecutive normal frames stacked along channel dimension.
    """
    def __init__(self, root_dir, file_list, json_file_path = None, stack_size=10, overlap=9, transform=None, only_normal=True):
        
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

        annotations_path = os.path.join(root_dir, "annotations")
        for json_file in json_files:
                json_path = os.path.join(annotations_path, json_file)
                if not os.path.exists(json_path):
                    print("JSON file doesnt exist")
                    continue

                with open(json_path, 'r') as f:
                    data = json.load(f)
                    normal_frames = [] 
                    for label in data["labels"]:
                        if only_normal and label["accident_name"] != "normal": # Skips anamolous frames
                            continue
                        normal_frames.append(os.path.join(root_dir, label["image_path"]))
                    
                    # generate stacks of consecutive frames
                    for i in range(0, len(normal_frames), _sliding_skip):
                        stack = normal_frames[i:i + stack_size]
                        # skip last incomplete stack
                        if len(stack) == stack_size:
                            self.stacks.append(stack)

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
        return stacked_tensor


# --------------------------------------- Helper Functionss -------------------------------------- #

# one validation run
def valid_epoch(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inp = batch.to(device)
            out = model(inp)
            loss = criterion(out, inp)
            val_loss += loss.item() * batch.size(0)
    val_loss /= len(val_loader.dataset)
    return val_loss


# --------------------------------------- Config Class --------------------------------------- #

class Config:
    SEED = 42

    # Data Directory -- Modify These
    ROOT = "/home/public/mkamal/datasets/deep_learning/projdata/uploaded_data"
    JSON_FILE_PATH = "/home/grad/masters/2025/mkamal/mkamal/dl_project/anamoly-detection/train_val.txt"

    # Image Transform Params
    IMAGE_SIZE = 224          # 224 works with this conv
    NORM_MEAN = [0.4333332]
    NORM_STD = [0.2551518]
    CONVERT_GRAY = 1

    # Dataset config
    IMAGE_STACK = 10           # 10 seem to provide the best results
    SLIDE_OVERLAP = 3          # sliding window overlap
    BATCH = 128
    NUM_WORKERS = 16

    # MODEL SETUP
    NUM_CHANNELS = CONVERT_GRAY

    # TRAINING SETUP
    EPOCH = 30
    LR = 2.4e-05     # Learning Rate 3e-06 -> 32,  
    DECAY = 3e-5   # Regularization Parameter
    EARLY_STOPPING_PATIENCE = 15   # Two epoch patience
    
    

def main():
    config = Config()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalization
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=config.CONVERT_GRAY),
        transforms.ToTensor(),
        transforms.Normalize(config.NORM_MEAN, config.NORM_STD)
    ])

    # ---------------------------- Dataset Setup ----------------------------
    # Train validation Split 90-10
    train_file_list, validation_file_list = split_before_training(config.JSON_FILE_PATH, 0.1, config.SEED)


    # Create datasets
    train_ds = StackedFramesDataset(
        config.ROOT, 
        file_list=train_file_list, 
        stack_size=config.IMAGE_STACK,
        overlap=config.SLIDE_OVERLAP, 
        transform=transform, 
        only_normal=True
    )

    val_ds = StackedFramesDataset(
        config.ROOT, 
        file_list=validation_file_list, 
        stack_size=config.IMAGE_STACK, 
        overlap=0,
        transform=transform, 
        only_normal=True
    )

    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=config.BATCH, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=config.BATCH, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    # ---------------------------- MODEL SETUP ---------------------------- #
    model = Conv2DAutoEncoder(config.NUM_CHANNELS*config.IMAGE_STACK).to(device) 
    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.DECAY)
    criterion = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    print("Initialized Model with ", sum(p.numel() for p in model.parameters()))
    
    # --------------------------- LR RANGE TEST AE ------------------------#

    # lr_range_test_autoencoder(model,train_loader, device, num_iter=250)

    # ---------------------------- TRAIN LOOP ---------------------------- #

    patience=config.EARLY_STOPPING_PATIENCE; best=float("inf"); waited=0; best_state=None

    tl_list=[]; vl_list=[]
    for epoch in range(config.EPOCH):
        print(f"Starting Epoch {epoch+1}")

        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc="Train Loader"):
            inp = batch.to(device)
            optimizer.zero_grad()
            out = model(inp)
            loss = criterion(out, inp)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_loader.dataset)
        tl_list.append(train_loss)

        # Validation calculation
        val_loss = valid_epoch(model, val_loader, device, criterion)
        vl_list.append(val_loss)

        print(f"Epoch: {epoch+1} || Train Loss: {train_loss} || Validation Loss: {val_loss}")

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

    # ---------------------------- PLOTTING ---------------------------- #

    save_path = "/home/public/mkamal/saved_models/convAE_best.pth"
    torch.save(best_state, save_path)
    print("Model saved in path: /home/public/mkamal/saved_models/convAE_best_model30.pth")

    plt.figure()
    plt.plot(tl_list, label="Train Loss")
    plt.plot(vl_list, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Metrics")
    plt.savefig("convAE_train_valid_loss_curve3.png")

if __name__ == "__main__":
    main()


    

    