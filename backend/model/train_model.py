
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

NUM_CLASSES = 62          # 10 digits + 26 uppercase + 26 lowercase
IMG_SIZE = 28
BATCH_SIZE = 256
EPOCHS = 1
MODEL_PATH = "hcr_model.keras"
LABEL_MAP_PATH = "label_map.json"

# EMNIST-ByClass label order: 0-9, then A-Z, then a-z
EMNIST_BYCLASS_LABELS = (
    list("0123456789")
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abcdefghijklmnopqrstuvwxyz")
)
assert len(EMNIST_BYCLASS_LABELS) == NUM_CLASSES


def load_emnist_byclass():
    """Loads EMNIST-ByClass (full case distinction) via tensorflow-datasets."""
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        raise SystemExit(
            "tensorflow-datasets is required to auto-download EMNIST.\n"
            "Install with: pip install tensorflow-datasets\n"
            "Alternatively, download the EMNIST-ByClass .mat/.csv files manually "
            "from https://www.nist.gov/itl/products-and-services/emnist-dataset "
            "and adapt load_from_csv() below."
        )

    ds_train, ds_test = tfds.load(
        "emnist/byclass", split=["train", "test"], as_supervised=True
    )

    def to_numpy(ds):
        images, labels = [], []
        for img, label in tfds.as_numpy(ds):
            images.append(img)
            labels.append(label)
        return np.array(images, dtype="float32"), np.array(labels, dtype="int64")

    x_train, y_train = to_numpy(ds_train)
    x_test, y_test = to_numpy(ds_test)

    # EMNIST images are stored transposed relative to normal reading orientation
    x_train = np.transpose(x_train, (0, 2, 1, 3))
    x_test = np.transpose(x_test, (0, 2, 1, 3))

    return x_train, y_train, x_test, y_test


def load_from_csv(train_csv, test_csv):
    """Alternative loader for manually downloaded EMNIST-ByClass CSV files."""
    train_data = np.loadtxt(train_csv, delimiter=",")
    test_data = np.loadtxt(test_csv, delimiter=",")

    y_train = train_data[:, 0].astype("int64")
    x_train = train_data[:, 1:].reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32")

    y_test = test_data[:, 0].astype("int64")
    x_test = test_data[:, 1:].reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32")

    return x_train, y_train, x_test, y_test


def build_model():
    """Deeper CNN with batch norm for better generalization across 62 classes."""
    model = Sequential([
        Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Flatten(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Loading EMNIST-ByClass (digits + upper + lower) ...")
    x_train, y_train, x_test, y_test = load_emnist_byclass()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    if x_train.ndim == 3:
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    # EMNIST-ByClass is naturally imbalanced (far fewer lowercase samples
    # than digits) — class weighting stops the model from just ignoring
    # rare classes, which is a common cause of "lowercase never predicted".
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)

    datagen = ImageDataGenerator(
        rotation_range=12,
        width_shift_range=0.12,
        height_shift_range=0.12,
        zoom_range=0.12,
        shear_range=8,
    )
    datagen.fit(x_train)

    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    print("Training ...")
    model.fit(
        datagen.flow(x_train, y_train_cat, batch_size=BATCH_SIZE),
        validation_data=(x_val, y_val_cat),
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )

    print("Evaluating on held-out test set ...")
    test_loss, test_acc = model.evaluate(x_test, y_test_cat)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(x_test), axis=1)
    print(classification_report(
        y_test, y_pred, target_names=EMNIST_BYCLASS_LABELS, zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    model.save(MODEL_PATH)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump({str(i): c for i, c in enumerate(EMNIST_BYCLASS_LABELS)}, f, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved label map to {LABEL_MAP_PATH}")


if __name__ == "__main__":
    main()