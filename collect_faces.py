"""Collect cropped face images from a webcam."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2

IMAGE_SIZE = (100, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", help="Person name used for the dataset directory.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


def safe_person_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    if not name:
        raise ValueError("Name must contain at least one letter or number.")
    return name


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be a positive integer.")

    person_name = safe_person_name(args.name or input("Enter the person's name: "))
    person_dir = args.dataset / person_name
    person_dir.mkdir(parents=True, exist_ok=True)

    existing_numbers = [int(path.stem) for path in person_dir.glob("*.jpg") if path.stem.isdigit()]
    next_number = max(existing_numbers, default=0) + 1

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector.empty():
        raise RuntimeError(f"Could not load face detector: {cascade_path}")

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    collected = 0
    print(f"Collecting {args.samples} images for '{person_name}'. Press Esc or Q to stop.")
    try:
        while collected < args.samples:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The camera stopped returning frames.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            if len(faces):
                x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
                face = gray[y : y + height, x : x + width]
                face = cv2.resize(face, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                output_path = person_dir / f"{next_number + collected}.jpg"
                if not cv2.imwrite(str(output_path), face):
                    raise OSError(f"Could not write image: {output_path}")
                collected += 1
                cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

            cv2.putText(
                frame,
                f"Captured: {collected}/{args.samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Collect Face Images", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Saved {collected} image(s) to {person_dir}")


if __name__ == "__main__":
    main()
