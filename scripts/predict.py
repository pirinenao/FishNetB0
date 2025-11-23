"""
This script loads a trained model and makes a single image prediction.
Default model path is "fishnetb0.keras".

Example usage:
python predict.py image.jpg
python predict.py -m path/to/model.keras image.jpg
"""

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import (
    preprocess_input,
)
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import sys
import os
import argparse

# configuration
DEFAULT_MODEL_PATH = "fishnetb0.keras"
IMG_SIZE = 224

# important: the class names must match the class names in the training dataset
CLASS_NAMES = [
    "Ahven",
    "Harjus",
    "Hauki",
    "Kiiski",
    "Kirjolohi",
    "Kuha",
    "Lahna",
    "Lohi",
    "Made",
    "Pasuri",
    "Sarki",
    "Sayne",
    "Siika",
    "Taimen",
]


# handle argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a single image prediction using a Keras model."
    )

    # required positional argument for the image path
    parser.add_argument(
        "image_path", type=str, help="Path to the image file you want to classify."
    )

    # optional argument for the model path
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the .keras model file. Defaults to '{DEFAULT_MODEL_PATH}'.",
    )

    return parser.parse_args()


# load and preprocess the image
def load_and_prep_image(image_path, target_size=(IMG_SIZE, IMG_SIZE)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)  # convert the image to a NumPy array
    img_batch = np.expand_dims(
        img_array, axis=0
    )  # add a "batch" dimension (from (224, 224, 3) to (1, 224, 224, 3))
    processed_img = preprocess_input(
        img_batch
    )  # preprocess the image (same as the training data)
    return processed_img


# --- Main script execution ---
if __name__ == "__main__":
    # parse arguments
    args = parse_arguments()
    image_path = args.image_path
    MODEL_PATH = args.model  # Use the path provided by the user or the default

    # check if the model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print(f"Specify a model using -m or ensure '{DEFAULT_MODEL_PATH}' exists.")
        sys.exit(1)

    # check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        sys.exit(1)

    # load the model
    try:
        print(f"Loading model from '{MODEL_PATH}'...")
        model = load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        sys.exit(1)

    # load and preprocess the image
    print(f"Loading and processing image: {image_path}")
    try:
        processed_image = load_and_prep_image(image_path)
    except Exception as e:
        print(f"An error occurred while processing the image: {e}")
        sys.exit(1)

    # make the prediction
    predictions = model.predict(
        processed_image, verbose=0
    )  # Use verbose=0 for clean output

    # interpret the results
    scores = predictions[0]
    predicted_index = np.argmax(scores)
    predicted_class_name = CLASS_NAMES[predicted_index]
    confidence = 100 * np.max(scores)

    # print the result
    print("\n--- Prediction Result ---")
    print(f"Model:      {MODEL_PATH}")
    print(f"Image:      {image_path}")
    print(f"Species:    **{predicted_class_name}**")
    print(f"Confidence: {confidence:.2f}%")

    # show all scores
    print("\n--- All Scores ---")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<10}: {scores[i] * 100:.2f}%")
