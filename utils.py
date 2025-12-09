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


"""
 Reads a text file of JSON filenames(made for) and returns them as a list
 Input: filepath ending with a .txt containng filenanmes
 output: list (file names)
"""
def read_json_files(json_file_path):
    if (not os.path.exists(json_file_path)):
        raise FileNotFoundError(f"No JSON file found at {json_file_path}")
    with open(json_file_path, "r") as f:
        json_files = [line.strip() for line in f.readlines()]
    
    return json_files



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


