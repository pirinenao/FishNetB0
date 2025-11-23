"""
This script was used to train the fish classification model using TensorFlow 2.20.0 and Keras 2.
The training process consists of two phases:
1. Training a new classification head with a frozen EfficientNetB0 base.
2. Fine-tuning the entire model (unfreezing the base) with a lower learning rate.
The script also includes data augmentation, model checkpointing, learning rate scheduling, and early stopping.
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt

# configuration
DATA_DIR = "Dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 14
EPOCHS_PHASE_1 = 10
EPOCHS_PHASE_2 = 90


def main():
    # check if data directory exists
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory not found at '{DATA_DIR}'")
        print("Please set DATA_DIR to the correct path.")
        return

    # load the datasets
    print("Loading datasets...")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )

    class_names = train_dataset.class_names
    print(f"Found classes: {class_names}")

    if len(class_names) != NUM_CLASSES:
        print(
            f"Error: Expected {NUM_CLASSES} classes, but found {len(class_names)} folders."
        )
        return

    # data pipeline optimizations
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # build the model
    print("Building model with Keras Applications EfficientNetB0")

    # 1. define the input layer
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

    # 2. data augmentation
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(factor=0.2),
            tf.keras.layers.RandomContrast(factor=0.2),
        ],
        name="data_augmentation",
    )

    x = data_augmentation(inputs)
    x = preprocess_input(x)

    # 4. create the base model
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # 5. freeze the base model for phase 1
    base_model.trainable = False

    # 6. pass the preprocessed data (x) through the base model
    x = base_model(x)

    # 7. pool the features
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # 8. add dropout (0.4)
    x = tf.keras.layers.Dropout(0.4, name="dropout")(x)

    # 9. add the final classification head
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="classification_head",
        kernel_regularizer=l2(0.001),
    )(x)

    # 10. create the final model
    model = tf.keras.Model(inputs, outputs, name="fishnetb0")

    model.summary()

    # callbacks
    checkpoint_cb = ModelCheckpoint(
        "fishnetb0_BEST.keras", save_best_only=True, monitor="val_loss", verbose=1
    )

    lr_scheduler_cb = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, verbose=1, min_lr=1e-7
    )

    early_stopping_cb = EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True, verbose=1
    )

    # bundle all callbacks
    CALLBACKS = [checkpoint_cb, lr_scheduler_cb, early_stopping_cb]

    # phase 1: training the new head (base frozen)
    print("\n--- Starting Phase 1: Training the new head (base frozen) ---")

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),  # legacy optimizer due to apple silicone compatibility issues with newer optimizers
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    history_phase_1 = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS_PHASE_1,
        callbacks=CALLBACKS,
    )

    # phase 2: fine-tuning (base un-frozen)
    print("\n--- Starting Phase 2: Fine-tuning the full model (base un-frozen) ---")

    base_model.trainable = True  # Unfreeze
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-5),  # custom LR
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    history_phase_2 = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS_PHASE_1 + EPOCHS_PHASE_2,
        initial_epoch=history_phase_1.epoch[-1] + 1,
        callbacks=CALLBACKS,
    )

    model.save("fishnetb0.keras")
    print("Successfully saved model to 'fishnetb0.keras'")

    # plot the results
    print("Training complete. Plotting results...")
    plot_history(history_phase_1, history_phase_2)


# plot the history
# combines and plots the history from two training phases
# using matplotlib
def plot_history(history1, history2):
    if not history2.history:
        history2_acc, history2_val_acc = [], []
        history2_loss, history2_val_loss = [], []
    else:
        history2_acc = history2.history["accuracy"]
        history2_val_acc = history2.history["val_accuracy"]
        history2_loss = history2.history["loss"]
        history2_val_loss = history2.history["val_loss"]

    acc = history1.history["accuracy"] + history2_acc
    val_acc = history1.history["val_accuracy"] + history2_val_acc
    loss = history1.history["loss"] + history2_loss
    val_loss = history1.history["val_loss"] + history2_val_loss

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.axvline(
        len(history1.history["accuracy"]) - 1,
        color="red",
        linestyle="--",
        label="Start Fine-Tuning",
    )
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.axvline(
        len(history1.history["loss"]) - 1,
        color="red",
        linestyle="--",
        label="Start Fine-Tuning",
    )
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_history.png")
    print("Saved training plot to 'training_history.png'")
    plt.show()


if __name__ == "__main__":
    main()
