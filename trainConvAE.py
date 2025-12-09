import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchvision import transforms

import copy
import matplotlib.pyplot as plt

from tqdm import tqdm

from ConvAE import Conv2DAutoEncoder
from datasetclasses import StackedFramesDataset


# ---------------------------- Dataset Setup ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "/home/public/mkamal/datasets/deep_learning/projdata/uploaded_data"
json_train_val_path = "/home/grad/masters/2025/mkamal/mkamal/dl_project/anamoly-detection/train_val.txt"

# Normal Frame Transform 224x224
transform_normal = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor([0.4193, 0.4373, 0.4470]), 
                         std=torch.tensor([0.2698, 0.2795, 0.2928]))
])

stack_size = 10
t_v_dataset = StackedFramesDataset(root,
                               stack_size=stack_size, 
                               overlap=0,
                               json_file_path=json_train_val_path,
                               transform=transform_normal,  
                               only_normal=True)

loader = DataLoader(t_v_dataset, batch_size=4, shuffle=True, num_workers=24, pin_memory=True)


# Train validation Split 90-10
num_samples = len(t_v_dataset)
train_size = int(0.9 * num_samples)
val_size   = num_samples-train_size

generator = torch.Generator().manual_seed(42)
train_ds, val_ds = random_split(t_v_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=24, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=24, pin_memory=True)

# ---------------------------- MODEL SETUP ---------------------------- #
EPOCHS = 5
model = Conv2DAutoEncoder(3*stack_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
# CosineAnnealingLR
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer,
#     T_max=EPOCHS,    # number of epochs to complete a cosine cycle
#     eta_min=1e-5 # minimum LR at the end of the cycle
# )
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ---------------------------- TRAIN LOOP ---------------------------- #

patience=2; best=float("inf"); waited=0; best_state=None

train_loss_list=[]; valid_loss_list=[]; val_acc=[]


for epoch in range(EPOCHS):
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
    train_loss_list.append(train_loss)

    # Validation calculation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation Loader"):
            inp = batch.to(device)
            out = model(inp)
            loss = criterion(out, inp)
            val_loss += loss.item() * batch.size(0)
    val_loss /= len(val_loader.dataset)
    valid_loss_list.append(val_loss)

    print(f"Epoch: {epoch+1} || Train Loss: {train_loss} || Validation Loss: {val_loss}")

    # Early Stopping logic
    if val_loss < best - 1e-4: 
        best=val_loss
        waited=0
        best_state=copy.deepcopy(model.state_dict())
    else:
        waited+=1
        if waited>patience:
            print("Early stopping.")
            break

# ---------------------------- PLOTTING ---------------------------- #

save_path = "/home/public/mkamal/saved_models/convAE_best_model.pth"
torch.save(best_state, save_path)

plt.figure()
plt.plot(train_loss_list, label="Train Loss")
plt.plot(valid_loss_list, label="Validation Loss")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Metrics")
plt.savefig("convAE_train_valid_loss_curve.png")



    

    