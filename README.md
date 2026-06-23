# Kvasir-SEG Multi U-Net Comparison

This repository is a medical image segmentation project for comparing several U-Net-style architectures on the **Kvasir-SEG** polyp segmentation dataset.

The goal is not only to train one model, but to keep the full workflow consistent across model variants so their validation metrics and qualitative predictions can be compared fairly.

## Project Idea

The project trains and compares four segmentation models:

- **U-Net**: the baseline encoder-decoder segmentation model.
- **Attention U-Net**: adds attention gates on skip connections.
- **U-Net++**: uses nested skip connections to reduce the semantic gap between encoder and decoder features.
- **TransUNet-style hybrid**: adds a Transformer encoder block at the bottleneck of a U-Net-like architecture.

All training notebooks use the same:

- dataset folder
- train/validation split seed
- image size
- batch size
- number of epochs
- early stopping settings
- metrics

This makes the comparison more meaningful.

## Dataset

The project uses **Kvasir-SEG**, a gastrointestinal endoscopy dataset for polyp segmentation.

Official dataset page:

https://datasets.simula.no/kvasir-seg/

The included code can download the dataset automatically from the current Simula download URL:

```text
https://datasets.simula.no/downloads/kvasir-seg.zip
```

In this project, the notebook workflow stores the dataset under:

```text
notebooks/data/Kvasir-SEG/
```

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_U_Net_Medical_Segmentation_Tutorial.ipynb
│   ├── 02_Attention_U_Net_Kvasir_SEG.ipynb
│   ├── 03_U_Net_PlusPlus_Kvasir_SEG.ipynb
│   ├── 04_TransUNet_Kvasir_SEG.ipynb
│   ├── 05_Model_Comparison_Kvasir_SEG.ipynb
│   ├── streamlit_model_viewer.py
│   ├── data/
│   │   └── Kvasir-SEG/
│   └── runs/
│       ├── unet/
│       ├── attention_unet/
│       ├── unet_plus_plus/
│       └── transunet/
└── src/
    ├── dataset.py
    ├── download_data.py
    ├── evaluate.py
    ├── metrics.py
    ├── models.py
    ├── predict.py
    ├── train.py
    └── visualize_samples.py
```

## Notebooks

### 1. U-Net Baseline

```text
notebooks/01_U_Net_Medical_Segmentation_Tutorial.ipynb
```

This is the baseline notebook. It downloads/loads the data, builds the dataset, defines the models, trains U-Net, evaluates Dice and IoU, and visualizes predictions.

### 2. Attention U-Net

```text
notebooks/02_Attention_U_Net_Kvasir_SEG.ipynb
```

This notebook has the same structure as the baseline notebook. The main difference is:

```python
MODEL_NAME = "attention_unet"
```

### 3. U-Net++

```text
notebooks/03_U_Net_PlusPlus_Kvasir_SEG.ipynb
```

This notebook has the same structure as the baseline notebook. The main difference is:

```python
MODEL_NAME = "unet_plus_plus"
```

### 4. TransUNet

```text
notebooks/04_TransUNet_Kvasir_SEG.ipynb
```

This notebook has the same structure as the baseline notebook. The main difference is:

```python
MODEL_NAME = "transunet"
```

### 5. Model Comparison and Streamlit GUI

```text
notebooks/05_Model_Comparison_Kvasir_SEG.ipynb
```

This notebook compares the trained checkpoints from:

```text
notebooks/runs/<model_name>/best.pt
```

It evaluates each model on the same validation split and includes a Streamlit interface where you can upload an image and view:

- each model prediction overlay
- each model probability map
- a voting overlay across all selected models
- a vote-count map showing how many models predicted each pixel as disease

The voting overlay becomes stronger where more models agree.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python` is not available on Windows, use `py`:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

## Training From the Command Line

Train the baseline U-Net:

```powershell
python -m src.train --model unet --download --epochs 30 --image-size 256 --batch-size 4 --max-samples None
```

Train Attention U-Net:

```powershell
python -m src.train --model attention_unet --epochs 30 --image-size 256 --batch-size 4 --max-samples None
```

Train U-Net++:

```powershell
python -m src.train --model unet_plus_plus --epochs 30 --image-size 256 --batch-size 4 --max-samples None
```

Train TransUNet:

```powershell
python -m src.train --model transunet --epochs 30 --image-size 256 --batch-size 4 --max-samples None
```

Training outputs are saved under:

```text
runs/<model_name>/
```

The notebook workflow saves checkpoints under:

```text
notebooks/runs/<model_name>/
```

## Early Stopping

Early stopping is enabled to reduce overfitting.

The training loop tracks validation Dice score. If validation Dice does not improve for the configured patience, training stops early.

Default settings:

```text
early_stop_patience = 5
early_stop_min_delta = 1e-4
```

## Metrics

The project reports:

- **Dice Score**: overlap between predicted mask and true mask.
- **IoU**: intersection over union.
- **Loss**: BCEWithLogitsLoss + Dice Loss.

For segmentation, Dice and IoU are more informative than plain pixel accuracy because the background usually covers most pixels.

## Streamlit Model Viewer

The Streamlit GUI is generated from the comparison notebook and saved at:

```text
notebooks/streamlit_model_viewer.py
```

Run it with:

```powershell
streamlit run notebooks/streamlit_model_viewer.py
```

or:

```powershell
python -m streamlit run notebooks/streamlit_model_viewer.py
```

The app lets you upload an image and compare all trained models visually.

## Prediction From the Command Line

Example:

```powershell
python -m src.predict --image notebooks/data/Kvasir-SEG/images/example.jpg --checkpoint notebooks/runs/unet/best.pt --output prediction.png
```

Replace `example.jpg` with an actual image filename from:

```text
notebooks/data/Kvasir-SEG/images/
```

## Notes

- The notebooks are designed for learning and comparison.
- The TransUNet implementation is a compact educational hybrid, not a reproduction of every detail from the original TransUNet paper.
- For a formal paper or production system, add stronger validation, cross-validation, more metrics, and reproducibility tracking.

## Citation

If you use Kvasir-SEG for research, cite the original dataset paper and review the dataset license/terms from the official Simula page.
