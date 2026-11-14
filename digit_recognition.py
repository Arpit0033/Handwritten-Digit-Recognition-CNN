
# ==========================================
# HANDWRITTEN DIGIT RECOGNITION USING CNN
# Advanced Data Augmentation Version
# ==========================================

# STEP 1: IMPORT LIBRARIES

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from tensorflow.keras import layers, models

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("TensorFlow Version:", tf.__version__)

# Create output directories

Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)


# ==========================================
# STEP 2: LOAD MNIST DATASET
# ==========================================

(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

print("\nDataset Loaded Successfully!")

print("Training Images:", X_train.shape)
print("Testing Images:", X_test.shape)


# ==========================================
# STEP 3: DISPLAY FIRST 10 IMAGES
# ==========================================

plt.figure(figsize=(12, 5))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(X_train[i], cmap="gray")

    plt.title(f"Label: {y_train[i]}")

    plt.axis("off")

plt.tight_layout()

plt.savefig("outputs/dataset_samples.png")

plt.show()
plt.close()


# ==========================================
# STEP 4: DATA PREPROCESSING
# ==========================================

# Normalize pixel values

X_train = X_train.astype("float32") / 255.0

X_test = X_test.astype("float32") / 255.0

# Reshape images for CNN

X_train = X_train.reshape(-1, 28, 28, 1)

X_test = X_test.reshape(-1, 28, 28, 1)

# Split training and validation datasets

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.10,
    random_state=42,
    stratify=y_train
)

print("\nAfter Preprocessing:")

print("Training Images:", X_train.shape)

print("Validation Images:", X_val.shape)

print("Testing Images:", X_test.shape)

print("Minimum Pixel Value:", X_train.min())

print("Maximum Pixel Value:", X_train.max())


# ==========================================
# STEP 5: ADVANCED DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([

    # Random Rotation
    layers.RandomRotation(
        factor=0.06,
        fill_mode="constant",
        fill_value=0
    ),

    # Random Translation
    layers.RandomTranslation(
        height_factor=0.08,
        width_factor=0.08,
        fill_mode="constant",
        fill_value=0
    ),

    # Random Zoom
    layers.RandomZoom(
        height_factor=(-0.10, 0.10),
        width_factor=(-0.10, 0.10),
        fill_mode="constant",
        fill_value=0
    )

], name="data_augmentation")


# ==========================================
# STEP 6: BUILD IMPROVED CNN MODEL
# ==========================================

model = models.Sequential([

    layers.Input(shape=(28, 28, 1)),

    # Data Augmentation
    data_augmentation,

    # First CNN Block
    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        use_bias=False
    ),

    layers.BatchNormalization(),

    layers.Activation("relu"),

    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Dropout(0.15),

    # Second CNN Block
    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        use_bias=False
    ),

    layers.BatchNormalization(),

    layers.Activation("relu"),

    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Dropout(0.20),

    # Third CNN Block
    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    # Flatten Layer
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(
        128,
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Dropout(0.35),

    # Output Layer
    layers.Dense(
        10,
        activation="softmax"
    )

])


# ==========================================
# STEP 7: COMPILE MODEL
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

print("\nCNN Model Summary:")

model.summary()


# ==========================================
# STEP 8: TRAINING CALLBACKS
# ==========================================

# Early Stopping

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1

)

# Automatically reduce learning rate

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=0.00001,

    verbose=1

)

# Save best model

checkpoint = ModelCheckpoint(

    "models/digit_cnn_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1

)


# ==========================================
# STEP 9: TRAIN CNN MODEL
# ==========================================

print("\nStarting Improved CNN Training...\n")

history = model.fit(

    X_train,

    y_train,

    epochs=25,

    batch_size=64,

    validation_data=(X_val, y_val),

    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint
    ],

    verbose=1

)

print("\nTraining Completed Successfully!")


# ==========================================
# STEP 10: LOAD BEST MODEL
# ==========================================

model = tf.keras.models.load_model(
    "models/digit_cnn_model.keras"
)

print("\nBest Model Loaded Successfully!")


# ==========================================
# STEP 11: PLOT ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Improved CNN Model Accuracy")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig(
    "outputs/accuracy_graph.png",
    bbox_inches="tight"
)

plt.show()
plt.close()


# ==========================================
# STEP 12: PLOT LOSS GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Improved CNN Model Loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig(
    "outputs/loss_graph.png",
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nAccuracy and Loss Graphs Saved!")


# ==========================================
# STEP 13: TEST MODEL PERFORMANCE
# ==========================================

test_loss, test_accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=1

)

print("\n========== FINAL TEST RESULTS ==========")

print(f"Test Loss: {test_loss:.4f}")

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


# ==========================================
# STEP 14: CONFUSION MATRIX
# ==========================================

print("\nGenerating Confusion Matrix...")

y_predictions = model.predict(

    X_test,

    verbose=1

)

y_predicted = np.argmax(

    y_predictions,

    axis=1

)

cm = confusion_matrix(

    y_test,

    y_predicted

)

fig, ax = plt.subplots(figsize=(10, 8))

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=np.arange(10)

)

disp.plot(

    ax=ax,

    cmap="Blues",

    values_format="d"

)

plt.title("Improved MNIST CNN Confusion Matrix")

plt.savefig(

    "outputs/confusion_matrix.png",

    bbox_inches="tight"

)

plt.show()
plt.close()

print("Confusion Matrix Saved!")


# ==========================================
# STEP 15: SAMPLE PREDICTIONS
# ==========================================

print("\nGenerating Sample Predictions...")

sample_predictions = model.predict(

    X_test[:10],

    verbose=0

)

plt.figure(figsize=(14, 7))

for i in range(10):

    predicted_digit = np.argmax(
        sample_predictions[i]
    )

    actual_digit = y_test[i]

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        X_test[i].squeeze(),
        cmap="gray"
    )

    status = (
        "Correct"
        if predicted_digit == actual_digit
        else "Wrong"
    )

    plt.title(
        f"Actual: {actual_digit}\n"
        f"Predicted: {predicted_digit}\n"
        f"{status}"
    )

    plt.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/sample_predictions.png",
    bbox_inches="tight"
)

plt.show()
plt.close()

print("Sample Predictions Saved!")


# ==========================================
# STEP 16: FINAL SUMMARY
# ==========================================

print("\n===================================")

print(" HANDWRITTEN DIGIT RECOGNITION")

print("===================================")

print("Model: Advanced CNN")

print("Dataset: MNIST")

print("Number of Classes: 10")

print(
    f"Final Test Accuracy: {test_accuracy * 100:.2f}%"
)

print("Model Path: models/digit_cnn_model.keras")

print("Accuracy Graph: outputs/accuracy_graph.png")

print("Loss Graph: outputs/loss_graph.png")

print("Confusion Matrix: outputs/confusion_matrix.png")

print("Sample Predictions: outputs/sample_predictions.png")

print("\nPROJECT TRAINING COMPLETED!")

print("===================================")
