# 📍 Mask Detection & Localization: Explainable AI with GAP

![Framework](https://img.shields.io/badge/Framework-TensorFlow_2.x-orange)
![Task](https://img.shields.io/badge/Task-Mask_Detection_%26_Localization-blue)
![Technique](https://img.shields.io/badge/Technique-Class_Activation_Maps_(CAM)-green)

This project implements an **Explainable Mask Detection System** using **Weakly Supervised Object Localization (WSOL)**. Unlike standard classifiers that simply output "Mask" or "No Mask," this model identifies the location of the face/mask in the image without ever being trained on bounding boxes.

By replacing traditional flattening layers with **Global Average Pooling (GAP)**, we can generate **Class Activation Maps (CAM)** to visualize exactly what the network is looking at—proving it is focusing on the face rather than background noise.

## 🎯 Project Goal
*   **Detection:** Classify images into binary categories: `WithMask` or `WithoutMask`.
*   **Localization:** Automatically generate heatmaps that highlight the facial region.
*   **Interpretability:** Use GAP + CAM to "open the black box" and verify the model's decision-making process.

## 🧠 Theory: From Classification to Localization
1.  **Standard CNN:** Typically ends with `Flatten` $\to$ `Dense` layers, which destroys spatial information.
2.  **Our Approach (GAP):** We use `GlobalAveragePooling2D` after the final convolution.
    *   The GAP layer averages each feature map into a single number.
    *   The final Dense layer learns weights for these averages.
3.  **CAM Generation:** By taking the weighted sum of the last convolutional feature maps (using the learned Dense weights), we reconstruct a heatmap.
    *   **For "WithMask":** The heatmap highlights the mask itself.
    *   **For "WithoutMask":** The heatmap highlights the facial features (nose, mouth).

## 📂 Dataset
*   **Domain:** Face Mask Detection.
*   **Classes:** Binary Classification (`WithMask`, `WithoutMask`).
*   **Input Size:** 256x256 pixels.
*   **Preprocessing:** Rescaling (`1./255`) and extensive data augmentation (rotation 10°, zoom 0.3, shifts) to prevent the model from memorizing background locations.

## 🏗️ Model Architecture
A custom VGG-style CNN designed for interpretability:
*   **Feature Extractor:** 5 blocks of `Conv2D` (3x3, ReLU) + `MaxPooling`.
*   **Filters:** Scaling from 8 $\to$ 16 $\to$ 32 $\to$ 64 $\to$ 128.
*   **The "Magic" Layer:** `GlobalAveragePooling2D` (output size: 128).
*   **Classifier:** `Dense` (2 units, Softmax).

## 📊 Results
*   **Classification:** Achieved **~99.5% Validation Accuracy** after just 7 epochs.
*   **Localization:** The generated heatmaps successfully localize the subject:
    *   Correctly focuses on the **mask region** when a mask is present.
    *   Correctly focuses on the **lower face** when no mask is present.

## 🛠️ Usage
1.  **Training:**
    Run the notebook to train the custom CNN on the Mask Dataset.
    ```python
    model.fit(train_dataset, validation_data=valid_dataset, epochs=20, callbacks=callbacks)
    ```
2.  **Visualization:**
    The notebook includes a custom function `generate_cam()` to overlay the heatmap on the original image:
    ```python
    # Visualizing where the model is looking
    heatmap = generate_cam(model, image, class_index)
    plot_heatmap(image, heatmap)
    ```

## 👨‍💻 Author
**Alex Miguel Franco Martínez**
