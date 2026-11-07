# Explainable Mask Detection with Class Activation Maps

A CNN that classifies faces as **with / without mask** and also shows **where** it is looking,
without ever being trained on bounding boxes (*weakly-supervised object localization*).

## Results

- **99.6% validation accuracy** after 7 epochs
- Lightweight model: **~98k parameters**
- Heatmaps (CAM) focus on the **mask** when present and on the **lower face** when absent,
  so the model is looking at the right thing, not at the background

## How it works

1. A VGG-style CNN (5 conv blocks, filters 8 → 128) extracts feature maps
2. Instead of `Flatten`, a **Global Average Pooling** layer keeps the spatial meaning of each map
3. The weights of the final `Dense` layer tell how much each feature map counts for a class
4. Their weighted sum gives a **Class Activation Map**, overlaid on the image as a heatmap

Training details: 256×256 input, data augmentation (rotation, shift, zoom, flips),
Adam, early stopping on validation loss.

## Stack

Python · TensorFlow / Keras · NumPy · Matplotlib

## Files

- [`Localization_GAP.ipynb`](Localization_GAP.ipynb): data pipeline, model, training, CAM visualisation
