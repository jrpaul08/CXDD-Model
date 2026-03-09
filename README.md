# Chest X-Ray Disease Detection (CXDD) Model

A multi-label classification model for detecting 14 thoracic diseases from chest X-ray images using transfer learning with ResNet50.

## Disease Classes

The model detects the following 14 conditions:

| Disease | Disease |
|---------|---------|
| Atelectasis | Mass |
| Cardiomegaly | Nodule |
| Consolidation | Pleural Thickening |
| Edema | Pneumonia |
| Effusion | Pneumothorax |
| Emphysema | |
| Fibrosis | |
| Hernia | |
| Infiltration | |

## Dataset

This project uses the **NIH Chest X-ray Dataset** containing 112,120 frontal-view X-ray images from 30,805 unique patients.

- **Training set:** ~89,696 images (80%)
- **Validation set:** ~22,424 images (20%)

### Downloading the Dataset

```bash
# Install Kaggle CLI
pip install kaggle

# Set up credentials (download kaggle.json from your Kaggle account settings)
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
kaggle datasets download -d nih-chest-xrays/data

# Extract images
unzip data.zip -d .
mv images_*/images/* images/
```

### Data Preprocessing

After downloading images, run the preprocessing cells in the notebook:

1. **Cell 7** - Creates `all_labels.csv` (binary label matrix from `Data_Entry_2017.csv`)
2. **Cell 8** - Creates train/val split CSVs (`train_labels.csv`, `val_labels.csv`)
3. **Cell 9** - Creates symlinks in `train/` and `val/` directories pointing to `images/`

## Project Structure

```
CXDD_model/
├── chest_xray.ipynb      # Main training notebook
├── requirements.txt      # Python dependencies
├── all_labels.csv        # Full dataset labels (binary matrix)
├── train_labels.csv      # Training split labels
├── val_labels.csv        # Validation split labels
├── dataset/              # Contains Data_Entry_2017.csv
│   └── Data_Entry_2017.csv
├── images/               # Raw X-ray images (not in repo)
├── train/                # Symlinks to training images (not in repo)
├── val/                  # Symlinks to validation images (not in repo)
└── README.md
```

## Environment Setup

### Requirements

- Python 3.9+
- CUDA-capable GPU (tested on NVIDIA H100)
- ~50GB disk space for images

### Installation

```bash
# Create virtual environment
python -m venv cxdd_env
source cxdd_env/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| torch | Deep learning framework |
| torchvision | Pre-trained models, transforms |
| pandas | Data manipulation |
| numpy | Numerical operations |
| scikit-learn | Evaluation metrics |
| pillow | Image loading |

## Workflow

The notebook walks through the following process:

1. **Download dataset** — Fetch images via Kaggle CLI
2. **Load data** — Read metadata and create binary label matrix
3. **Preprocess data** — Train/val split and create symlinks
4. **Train model** — Transfer learning with early stopping and checkpointing
5. **Evaluate** — Metrics and sample predictions on validation set

## Model Weights

Model weights are not included in this repository due to size (~95MB each). 

To obtain trained weights:
- **Train from scratch** using the notebook
- **Download from Releases** (if available)

## License

The NIH Chest X-ray dataset is provided under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) license.

## References

- Wang, X., et al. "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks." CVPR 2017.
- [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [Kaggle Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)
