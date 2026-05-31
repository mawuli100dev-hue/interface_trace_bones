"""
engine.py — Core ML logic: preprocessing, inference, training subprocess bridge.
"""

import os
import sys
import subprocess
import threading

import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

# ─── Constants ───────────────────────────────────────────────────────────────

IMG_WIDTH, IMG_HEIGHT = 200, 250
CONFIDENCE_THRESHOLD = 0.7

CLASS_NAMES_4 = ["crocodile", "hyène", "léopard", "lion"]
CLASS_NAMES_5 = ["crocodile", "hyène", "léopard", "lion", "lycaons"]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ─── Image helpers ────────────────────────────────────────────────────────────

def preprocess_image(img_path: str) -> np.ndarray:
    """Load, convert to grayscale-RGB, resize and ResNet-preprocess an image."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
    img_pre = tf.keras.applications.resnet.preprocess_input(img_resized.astype(np.float32))
    return np.expand_dims(img_pre, axis=0)


def classify_image(model, class_names: list[str], img_path: str) -> tuple[str, float]:
    """Return (predicted_class, confidence) for a single image path."""
    tensor = preprocess_image(img_path)
    preds = model.predict(tensor, verbose=0)
    idx = int(np.argmax(preds))
    confidence = float(preds[0][idx])
    if confidence < CONFIDENCE_THRESHOLD:
        return "inconnu", confidence
    return class_names[idx], confidence


# ─── Model helpers ────────────────────────────────────────────────────────────

def load_model(path: str):
    """Load a Keras .h5 model and return (model, class_names)."""
    model = tf.keras.models.load_model(path)
    n_classes = model.output_shape[-1]
    class_names = CLASS_NAMES_5 if n_classes == 5 else CLASS_NAMES_4
    return model, class_names


# ─── Batch classification ─────────────────────────────────────────────────────

def classify_paths(model, class_names: list[str], paths: list[str]) -> list[dict]:
    """Classify a list of file paths; return list of result dicts."""
    results = []
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        try:
            pred, conf = classify_image(model, class_names, path)
            results.append({
                "Fichier": os.path.basename(path),
                "Classe": pred,
                "Confiance": f"{conf * 100:.2f}%",
                "_conf_raw": conf,
            })
        except Exception as e:
            results.append({
                "Fichier": os.path.basename(path),
                "Classe": "Erreur",
                "Confiance": "0%",
                "_conf_raw": 0.0,
            })
    return results


def classify_folder(model, class_names: list[str], folder: str) -> list[dict]:
    """Classify all supported images in a directory."""
    paths = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.isfile(os.path.join(folder, f))
    ]
    return classify_paths(model, class_names, paths)


# ─── Statistics ───────────────────────────────────────────────────────────────

def compute_stats(results: list[dict]) -> dict[str, int]:
    """Return count per class label from a results list."""
    counts: dict[str, int] = {}
    for r in results:
        lbl = r["Classe"]
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert results list to a clean DataFrame (no internal keys)."""
    rows = [{"Fichier": r["Fichier"], "Classe": r["Classe"], "Confiance": r["Confiance"]} for r in results]
    return pd.DataFrame(rows, columns=["Fichier", "Classe", "Confiance"])


# ─── Training subprocess bridge ───────────────────────────────────────────────

def launch_training(
    train_dir: str,
    val_dir: str,
    save_path: str,
    epochs: int,
    device: str,
    on_line,          # callback(line: str)
    on_progress,      # callback(pct: float)
    on_done,          # callback(returncode: int)
):
    """
    Spawn training.py in a background thread.
    Calls on_line(line) for each output line.
    Calls on_progress(pct) when an epoch progress line is detected.
    Calls on_done(returncode) when the process ends.
    """
    cmd = [
        sys.executable, "training.py",
        "--train_dir", train_dir,
        "--validation_dir", val_dir,
        "--save_path", save_path,
        "--epochs", str(epochs),
        "--device", device,
    ]

    def _run():
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in iter(process.stdout.readline, ""):
                on_line(line)
                # Parse "Epoch X/Y" progress lines
                if "Epoch" in line and "/" in line:
                    try:
                        parts = line.strip().split("/")
                        current = int(parts[0].split()[-1])
                        total = int(parts[1].split()[0])
                        on_progress(current / total * 100)
                    except (ValueError, IndexError):
                        pass
            process.stdout.close()
            returncode = process.wait()
            on_done(returncode)
        except Exception as exc:
            on_line(f"[ERREUR] {exc}\n")
            on_done(-1)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread
