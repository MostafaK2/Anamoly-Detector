import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import json
from PIL import Image
from torchvision.io import read_image
import cv2

from utils import read_json_files

class MeanVideoFramesDataset(Dataset):
    def __init__(self, root_dir, json_file_path, transform=None, only_normal=True):

        assert os.path.exists(json_file_path), f"JSON list file not found: {json_file_path}"
        
        # Read JSON filenames from the text file
        json_files = read_json_files(json_file_path)

        self.root_dir = root_dir
        self.transform = transform
        self.only_normal = only_normal
        self.image_paths = []

        annotations_path = os.path.join(root_dir, "annotations")
        for json_file in json_files:
            json_path = os.path.join(annotations_path, json_file)
            if not os.path.exists(json_path):
                print("JSON file doesnt exist")
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)
                for label in data["labels"]:
                    if only_normal and label["accident_name"] != "normal": #if its a anamolous segment
                        continue
                    self.image_paths.append(os.path.join(root_dir, label["image_path"]))
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
