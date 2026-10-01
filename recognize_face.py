"""Recognize faces from a webcam using a trained CNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from keras.models import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("face_model.keras"))
    parser.add_argument("--labels", type=Path, default=Path("face_labels.json"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Minimum softmax confidence required to display a known name.",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> tuple[list[str], tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Label metadata not found: {path}. Run train_model.py first.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    class_names = metadata.get("class_names")
    image_size = metadata.get("image_size")
    if not class_names or not isinstance(class_names, list):
        raise ValueError(f"Invalid class_names in {path}")
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise ValueError(f"Invalid image_size in {path}")
    return class_names, (int(image_size[0]), int(image_size[1]))


def main() -> None:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1.")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}. Run train_model.py first.")

    class_names, image_size = load_metadata(args.labels)
    model = load_model(args.model)
    if model.output_shape[-1] != len(class_names):
        raise ValueError("Model output count does not match the saved label metadata.")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector.empty():
        raise RuntimeError(f"Could not load face detector: {cascade_path}")

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    print("Recognition started. Press Esc or Q to quit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The camera stopped returning frames.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for x, y, width, height in faces:
                face = gray[y : y + height, x : x + width]
                face = cv2.resize(face, image_size, interpolation=cv2.INTER_AREA)
                face_input = face.astype(np.float32)[np.newaxis, ..., np.newaxis] / 255.0

                probabilities = model.predict(face_input, verbose=0)[0]
                label_index = int(np.argmax(probabilities))
                confidence = float(probabilities[label_index])
                name = class_names[label_index] if confidence >= args.threshold else "Unknown"
                color = (0, 180, 0) if name != "Unknown" else (0, 165, 255)
                caption = f"{name} ({confidence:.0%})"

                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(
                    frame,
                    caption,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            cv2.imshow("CNN Face Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
