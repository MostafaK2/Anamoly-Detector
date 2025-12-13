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
                    print(json_path)
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



class StackedFramesDatasetTest(Dataset):
    """
    Dataset that returns 10 consecutive normal frames stacked along channel dimension.
    
    """
    def __init__(self, root_dir, file_list, json_file_path = None, stack_size=10, overlap=0, transform=None, only_normal=False):
        
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
        self.stack_labels = []

        annotations_path = os.path.join(root_dir, "annotations")
        for json_file in json_files:
                json_path = os.path.join(annotations_path, json_file)
                if not os.path.exists(json_path):
                    print(json_path)
                    print("JSON file doesnt exist")
                    continue
            
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    frames = []
                    frame_indices = []

                    anomaly_start = data.get("anomaly_start", None)
                    anomaly_end = data.get("anomaly_end", None)

                    for label in data["labels"]:
                        if only_normal and label["accident_name"] != "normal": # Skips anamolous frames
                            continue

                        frame_idx = label["frame_id"]
                        frames.append(os.path.join(root_dir, label["image_path"]))
                        frame_indices.append(frame_idx)
                    
                    # generate stacks of consecutive frames
                    # Generate stacks of consecutive frames
                    for i in range(0, len(frames), _sliding_skip):
                        stack = frames[i:i + stack_size]
                        stack_frame_indices = frame_indices[i:i + stack_size]
                        
                        # Skip last incomplete stack
                        if len(stack) != stack_size:
                            continue
                        
                        # Determine if this stack contains anomalous frames
                        is_anomalous = False
                        
                        if anomaly_start is not None and anomaly_end is not None:
                            # Check if ANY frame in the stack falls within anomaly range
                            for frame_idx in stack_frame_indices:
                                if anomaly_start <= frame_idx <= anomaly_end:
                                    is_anomalous = True
                                    break
                        
                        # If only_normal is True, skip anomalous stacks
                        if only_normal and is_anomalous:
                            continue
                        
                        # Add stack and its label
                        self.stacks.append(stack)
                        self.stack_labels.append(1 if is_anomalous else 0)
            
        print(f"Generated {len(self.stacks)} stacks of 10 frame clips, where the label distribution is: Normal={sum(1 for l in self.stack_labels if l == 0)}, Anomalous={sum(1 for l in self.stack_labels if l == 1)}")

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
        return stacked_tensor, self.stack_labels[idx]


