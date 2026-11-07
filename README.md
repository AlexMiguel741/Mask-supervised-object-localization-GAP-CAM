# Weakly-Supervised Object Localization with GAP and Class Activation Maps

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Task](https://img.shields.io/badge/Task-Classification%20%2B%20Localization-blue)
![XAI](https://img.shields.io/badge/XAI-Class%20Activation%20Maps-green)

## Overview

Binary image classifier (`WithMask` / `WithoutMask`) that also **localizes the discriminative
region** of the image, trained with image-level labels only (no bounding boxes).
Localization is obtained through **Global Average Pooling** and **Class Activation Maps (CAM)**,
which also make the model's decisions interpretable.

## Key Results

| Metric | Value |
| :-- | :-- |
| Validation accuracy | **99.6%** |
| Epochs to converge | 7 |
| Parameters | ~98k |
| Localization | CAM peaks on the mask region (`WithMask`) and lower face (`WithoutMask`) |

## Model Architecture

```
Input 256×256×3
 └─ 5 × [Conv2D 3×3 + ReLU ─► MaxPooling 2×2]     filters: 8 → 16 → 32 → 64 → 128
 └─ GlobalAveragePooling2D                         128-d, preserves per-map spatial meaning
 └─ Dense(2, softmax)
```

**CAM computation:** for class *c*, the heatmap is the weighted sum of the last convolutional
feature maps, using the weights of the final dense layer for *c*:
`CAM_c(x, y) = Σ_k w_k^c · f_k(x, y)`, upsampled and overlaid on the input image.

## Training Setup

| | |
| :-- | :-- |
| Data split | 10,000 train · 800 validation · 992 test |
| Input | 256 × 256 RGB, rescaled to [0, 1] |
| Augmentation | rotation ±10°, width/height shift, zoom 0.3, horizontal and vertical flip |
| Optimizer | Adam, lr = 1e-3 |
| Loss | Categorical cross-entropy |
| Batch size | 32 |
| Callbacks | Early stopping on validation loss (patience 10), checkpointing |

## Tech Stack

Python · TensorFlow / Keras · NumPy · Matplotlib

## Repository Structure

```
├── Localization_GAP.ipynb   # Data pipeline, model, training, CAM generation and visualization
└── README.md
```

## Run

Open `Localization_GAP.ipynb` in Jupyter or Google Colab (GPU recommended) and run all cells.
