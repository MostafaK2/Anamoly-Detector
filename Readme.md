# Anomaly-Detector

Anomaly Detection in Egocentric Traffic Videos using Deep Learning

## Project Overview
This project explores two aproaches in detecting anamolies in ego-centric traffic videos
1. **Unsupervised Approach**: Convolutional Autoencoder (ConvAE) - reconstruction-based anomaly detection
2. **Supervised Approach**: VideoMAE + Classifier - fine-tuned transformer for binary classification

The models are trained and evaluated on the DoTA dataset for egocentric driving scenarios.

## Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installations
1. Clone the repository:
```bash
git clone https://github.com/MostafaK2/Anamoly-Detector.git
cd Anamoly-Detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
opencv-python
Pillow
numpy
matplotlib
scikit-learn
tqdm
```

## Dataset
#### Download DoTA dataset
- Find dataset in this drive folder ```https://drive.google.com/drive/folders/1_WzhwZC2NIpzZIpX7YCvapq66rtBc67n```
- Command to download ```gdown https://drive.google.com/uc?id=FILE_ID```
- Command to export ```7z x DoTA_seg.7z```
#### Generate text files
- First generaate .txt file for seperating the train/validation/test using functions split_by_json_file, and scan_and_export_json_files from utils.py
- Currently filenames already uploaded into the repository (generate if you would like)
#### For Videomae + Classifier
1. Update paths in the scripts to point to your dataset location and output directory for generated clips
2. Preprocess the dataset (for faster training):
```bash
python scripts_preprocessing.py
```
3. Currently commented out

This will create preprocessed clips in:
- `preprocessed_clips/train/`
- `preprocessed_clips/validation/`
- `preprocessed_clips/test/`

## Training

### 1. Training ConvAE (Unsupervised)
```bash
cd ConvAE
python ConvAE_pipeline.py
```

**Configuration options** (edit in `trainConvAE.py`):
- `BATCH`: Batch size (default: 32)
- `EPOCH`: Number of epochs (default: 30)
- `LR`: Learning rate (default: 1e-4)
- `STACK_SIZE`: Number of frames per sequence (default: 16)
- `OVERLAP`: Frame overlap between sequences (default: 8)

### 2. Training VideoMAE Classifier (Supervised)
```bash
cd videomae
python trainVideoMAEAnamoly.py
```

**Configuration options** (edit in `Config` class):
- `BATCH`: Batch size (default: 64)
- `EPOCH`: Number of epochs (default: 6)
- `LR`: Learning rate (default: 6.2e-4)
- `FREEZE_LAYERS`: Number of frozen encoder layers (default: 9)
- `DROPOUT`: Dropout rate (default: 0.5)
- `EARLY_STOPPING_PATIENCE`: Early stopping patience (default: 2)

**Important paths to update:**
- `ROOT`: Dataset root directory
- `PREPROCESSED_TRAIN`: Path to preprocessed training data
- `PREPROCESSED_VALID`: Path to preprocessed validation data

## Evaluation & Visualization

### Evaluate ConvAE and VideoMAE + classifier
```bash
cd ConvAE
python evaluateConvAE.py
python visualizeConvAE.py
python testVideoMAEAnamoly.py
python _videomae_visualize_results.py
```
Note: These codes will use configurations from ConvAE_pipeline.py code

Outputs:
- AUC-ROC score
- Reconstruction error plots
- Anomaly score distributions
- Visualization Directory
** Important paths to update **
- `ROOT`: Path where the actual raw dataset is located
- `MODEL_PATH`: Pretrained model path
- 'TEST_TXT_PATH': File containing JSON filenames for test videos
- `OUTPUT_DIR`: Specify where you would like to output ur evaluation or visualization results

These would create
- Evaluation results text files for both convae and videomae+classifier
- (Conv-AE) Reconstruction error plots
- (video-mae) TP, FP, FN, TN examples with frames
- (video-mae) Score distributions over the computed sequences with frames
- (video-mae) Anomaly scores over time with ground truth

## Results and Visualization
#### MODEL Performance

|   Model  | AUC-ROC  |
|----------|----------|
| ConvAE   |  0.556   |
| VideoMAE |  **0.828**   | 


#### Videomae+classifier further evaluation
| Class | Precision | Recall | F1-Score | Support | 
|-------|----------|----------|----------|----------|
| Normal       | 0.7437 |0.8573 | 0.7965 | 1472
| Anomaly      | 0.8094 | 0.6722 | 0.7345 | 1327
| Weighted Avg | 0.775 | 0.769 | 0.767 | 2799

## Acknowledgments
- VideoMAE pretrained model from [MCG-NJU](https://github.com/MCG-NJU/VideoMAE)
- DoTA Dataset for traffic anomaly detection
- Hugging Face Transformers library
