"""Train the CNN used by the webcam face-recognition demo."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    RandomFlip,
    RandomRotation,
    RandomZoom,
)
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (100, 100)
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--model", type=Path, default=Path("face_model.keras"))
    parser.add_argument("--labels", type=Path, default=Path("face_labels.json"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_names = sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())
    if len(class_names) < 2:
        raise ValueError("The dataset must contain at least two person directories.")

    images: list[np.ndarray] = []
    labels: list[int] = []
    skipped = 0

    for label_index, person_name in enumerate(class_names):
        person_dir = dataset_dir / person_name
        image_paths = sorted(
            path
            for path in person_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if len(image_paths) < 2:
            raise ValueError(f"'{person_name}' needs at least two readable images.")

        loaded_for_person = 0
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                skipped += 1
                continue
            images.append(cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA))
            labels.append(label_index)
            loaded_for_person += 1

        if loaded_for_person < 2:
            raise ValueError(f"'{person_name}' has fewer than two readable images.")

    if skipped:
        print(f"Warning: skipped {skipped} unreadable image(s).")

    data = np.asarray(images, dtype=np.float32)[..., np.newaxis] / 255.0
    targets = np.asarray(labels, dtype=np.int32)
    return data, targets, class_names


def build_model(class_count: int) -> Sequential:
    return Sequential(
        [
            Input(shape=(*IMAGE_SIZE, 1)),
            RandomFlip("horizontal"),
            RandomRotation(0.05),
            RandomZoom(0.1),
            Conv2D(32, 3, activation="relu"),
            MaxPooling2D(),
            Conv2D(64, 3, activation="relu"),
            MaxPooling2D(),
            Conv2D(128, 3, activation="relu"),
            GlobalAveragePooling2D(),
            Dense(128, activation="relu"),
            Dropout(0.35),
            Dense(class_count, activation="softmax"),
        ],
        name="face_recognition_cnn",
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive integers.")
    if not 0 < args.validation_split < 0.5:
        raise ValueError("--validation-split must be greater than 0 and less than 0.5.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data, targets, class_names = load_dataset(args.dataset)
    x_train, x_validation, y_train, y_validation = train_test_split(
        data,
        targets,
        test_size=args.validation_split,
        random_state=args.seed,
        stratify=targets,
    )
    y_train = to_categorical(y_train, num_classes=len(class_names))
    y_validation = to_categorical(y_validation, num_classes=len(class_names))

    print(f"Loaded {len(data)} images across {len(class_names)} people: {', '.join(class_names)}")
    print(f"Training images: {len(x_train)} | Validation images: {len(x_validation)}")

    model = build_model(len(class_names))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
        verbose=2,
    )

    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.labels.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model)
    metadata = {
        "class_names": class_names,
        "image_size": list(IMAGE_SIZE),
        "color_mode": "grayscale",
    }
    args.labels.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved model to {args.model}")
    print(f"Saved label metadata to {args.labels}")


if __name__ == "__main__":
    main()
