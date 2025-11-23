# FishNetB0

An image classification model based on Google's EfficientNetB0, trained on a custom dataset of over 6,000 images of common finnish fish species. Lightweight size (approximately 17MB) makes it suitable for web and mobile applications.

Access the web demo [here](https://pirinenao.github.io/FishNetB0/).


## Training Method

The model was trained using a two-phase transfer learning approach.

1.  **Phase 1 (Feature Extraction):** The pre-trained EfficientNetB0 base (ImageNet weights) was frozen, and only a custom top classification head was trained for 10 epochs.
2.  **Phase 2 (Fine-Tuning):** The entire model was un-frozen and fine-tuned with a very low learning rate to adapt the features.

**Training configuration:**

*   **Base Model:** EfficientNetB0 (ImageNet weights)
*   **Augmentation:** Random flip, rotation, zoom, brightness, and contrast.
*   **Regularization:** Dropout (0.4) and L2 Regularization to prevent overfitting.
*   **Callbacks:** ModelCheckpoint (save best), ReduceLROnPlateau (adaptive learning rate), and EarlyStopping.

>**_NOTE:_**  The model was trained using TensorFlow 2.20.0 and Keras 2 with the legacy optimizer, required for Apple Silicon compatibility. If you encounter import issues, set TF_USE_LEGACY_KERAS=True to force TensorFlow to use the legacy Keras version.

**Dataset:**

The model was trained on a custom collection of more than 6,000 labeled fish images. All photos were taken after the fish were caught, often on a surface or while being held by a person. No underwater images were included, so the **model performs best on fish photographed out of the water.**

**Results:**

The current best model performance was achieved at epoch 53, with:
*   **Validation Loss:** 0.63
*   **Validation Accuracy:** 82%

## Trained Species

The model supports the following 14 species:

| Finnish Name | English Name / Description |
| :--- | :--- |
| **Ahven** | Perch |
| **Harjus** | Grayling |
| **Hauki** | Northern Pike |
| **Kiiski** | Ruffe |
| **Kirjolohi** | Rainbow Trout |
| **Kuha** | Zander (Pike-perch) |
| **Lahna** | Bream |
| **Lohi** | Salmon |
| **Made** | Burbot |
| **Pasuri** | Silver Bream |
| **Särki** | Roach |
| **Säyne** | Ide |
| **Siika** | Whitefish |
| **Taimen** | Trout |

## Example usage

You can use the scrips/predict.py to test the model with your own images.

```bash
pip install -r requirements.txt
python predict.py your_test_img.jpg
```

## Conversion to TensorFlow.js

The model was converted to the TensorFlow.js format for the web demo by using the tensorflowjs_converter (python 3.11.8).
If you don't want to convert the model, you can copy the content of docs/web_model_ouput and start building with the model.

```bash
tensorflowjs_converter --input_format=keras --output_format=tfjs_graph_model fishnetb0.keras ./web_model_output
```
