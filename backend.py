# backend.py
import os
import time
import json
import logging
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms
from facenet_pytorch import MTCNN

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "Dense121_Net.pth"   # change as needed
NUM_CLASSES = 4
CLASS_NAMES = ['Clear Skin', 'Dark Spots', 'Puffy Eyes', 'Wrinkles']
IMG_SIZE = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_DIR = Path("results")
LOG_FILE = "predictions.log"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# LOGGER
# -------------------------
logger = logging.getLogger("inference_logger")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
if not logger.handlers:
    logger.addHandler(fh)

# -------------------------
# TRANSFORMS
# -------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------
# MODEL LOADER (flexible)
# -------------------------
def build_efficientnet(num_classes: int = NUM_CLASSES):
    """Create EfficientNet-B0 and replace classifier head."""
    try:
        base = models.efficientnet_b0(weights='IMAGENET1K_V1')
    except Exception:
        base = models.efficientnet_b0(pretrained=True)
    try:
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
    except Exception:
        in_features = base.classifier.in_features
        base.classifier = nn.Linear(in_features, num_classes)
    return base

def load_model(model_path: str = MODEL_PATH, num_classes: int = NUM_CLASSES):
    """Load model either full or via state_dict."""
    model_path = Path(model_path)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(model_path)

    try:
        model = torch.load(str(model_path), map_location=DEVICE)
        if isinstance(model, dict):
            raise ValueError("State dict detected, switching to architecture + load_state_dict.")
        model.to(DEVICE)
        model.eval()
        logger.info(f"Loaded full model from {model_path}")
        return model
    except Exception:
        try:
            state = torch.load(str(model_path), map_location=DEVICE)
            model = build_efficientnet(num_classes=num_classes)
            model.load_state_dict(state)
            model.to(DEVICE)
            model.eval()
            logger.info(f"Built EfficientNet and loaded state_dict from {model_path}")
            return model
        except Exception as e:
            logger.exception("Failed loading model (full or state_dict).")
            raise RuntimeError("Failed to load model file.") from e

# -------------------------
# MTCNN (cached)
# -------------------------
_mtcnn = None
def get_mtcnn(device=DEVICE):
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(keep_all=True, device=device)
    return _mtcnn

# -------------------------
# AGE MAP (Randomized single-age lists)
# -------------------------
AGE_MAP = {
    'Wrinkles': list(range(50, 71)),   # 50–70
    'Dark Spots': list(range(40, 51)), # 40–50
    'Puffy Eyes': list(range(30, 41)), # 30–40
    'Clear Skin': list(range(20, 31))  # 20–30
}

# -------------------------
# PREDICTION / INFERENCE
# -------------------------
def preprocess_pil(face_pil: Image.Image):
    """Apply transforms and return batched tensor on device."""
    return transform(face_pil).unsqueeze(0).to(DEVICE)

def predict_tensor(model: torch.nn.Module, tensor: torch.Tensor) -> Tuple[np.ndarray, int]:
    """Return probabilities (np array) and predicted index."""
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return probs, pred_idx

def detect_and_predict(model: torch.nn.Module, pil_image: Image.Image, class_names: List[str] = CLASS_NAMES) -> Dict[str, Any]:
    """Detect faces, run predictions, assign random age, annotate and return results."""
    mtcnn = get_mtcnn()
    boxes, _ = mtcnn.detect(pil_image)
    result = {"faces": [], "annotated_image_path": None, "processing_time": None}
    start = time.perf_counter()

    if boxes is None:
        result["processing_time"] = time.perf_counter() - start
        return result

    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1i, y1i, x2i, y2i = map(int, [max(0, x1), max(0, y1),
                                       min(pil_image.width, x2), min(pil_image.height, y2)])
        face_crop_pil = Image.fromarray(cv2.cvtColor(
            img_cv[y1i:y2i, x1i:x2i], cv2.COLOR_BGR2RGB))
        if face_crop_pil.width == 0 or face_crop_pil.height == 0:
            continue

        inp = preprocess_pil(face_crop_pil)
        probs, pred_idx = predict_tensor(model, inp)
        label = class_names[pred_idx]
        confidence = float(probs[pred_idx])

        # ✅ Pick a single random age (no array stored or returned)
        predicted_age = random.choice(AGE_MAP.get(label, list(range(18, 60))))

        # Store clean result
        face_entry = {
            "face_id": i + 1,
            "box": [x1i, y1i, x2i, y2i],
            "label": label,
            "confidence": confidence,
            "age": predicted_age  # only this value
        }
        result["faces"].append(face_entry)

        # Draw rectangle + label + age
        cv2.rectangle(img_cv, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        text = f"{label} ({confidence*100:.1f}%)  Age: {predicted_age}"
        cv2.putText(img_cv, text, (x1i, max(15, y1i - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save annotated image
    timestamp = int(time.time() * 1000)
    out_path = RESULTS_DIR / f"annotated_{timestamp}.jpg"
    cv2.imwrite(str(out_path), img_cv)
    elapsed = time.perf_counter() - start

    result["annotated_image_path"] = str(out_path)
    result["processing_time"] = elapsed

    # Log cleanly (no arrays)
    log_entry = {
        "timestamp": int(time.time()),
        "model_path": str(MODEL_PATH),
        "processing_time_s": elapsed,
        "num_faces": len(result["faces"]),
        "faces": result["faces"]
    }
    logger.info(json.dumps(log_entry))
    return result
