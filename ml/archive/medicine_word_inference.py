from pathlib import Path
import csv
import json

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
import torchvision.transforms as transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEGMENT_ROOT = PROJECT_ROOT / "ml/results/ocr_ml003/medicine_word_segments_p1"
MANIFEST = SEGMENT_ROOT / "medicine_word_segments_p1.csv"
ONNX_MODEL = PROJECT_ROOT / "ml/results/optimization/ML003_resnet18.onnx"
CLASS_MAPPING = PROJECT_ROOT / "ml/config/class_mapping.json"
OUTPUT = PROJECT_ROOT / "ml/results/ocr_ml003/medicine_word_inference_results.csv"

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256


class ResizeWithPadding:
    def __init__(self, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
        self.height = height
        self.width = width

    def __call__(self, image):
        image = image.convert("L")
        original_width, original_height = image.size
        scale = min(self.width / original_width, self.height / original_height)
        new_width = max(1, int(original_width * scale))
        new_height = max(1, int(original_height * scale))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        canvas = Image.new("L", (self.width, self.height), 255)
        x = (self.width - new_width) // 2
        y = (self.height - new_height) // 2
        canvas.paste(image, (x, y))
        return canvas


TRANSFORM = transforms.Compose([
    ResizeWithPadding(),
    transforms.ToTensor(),
])


def load_class_mapping():
    with open(CLASS_MAPPING, encoding="utf-8") as f:
        data = json.load(f)

    idx_to_class = data.get("idx_to_class")
    if idx_to_class is None:
        class_to_idx = data.get("class_to_idx", {})
        idx_to_class = {str(idx): name for name, idx in class_to_idx.items()}

    return {int(idx): name for idx, name in idx_to_class.items()}


def load_session():
    if not ONNX_MODEL.exists():
        raise FileNotFoundError(f"ONNX model not found: {ONNX_MODEL}")

    return ort.InferenceSession(
        str(ONNX_MODEL),
        providers=["CPUExecutionProvider"],
    )


def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")
    tensor = TRANSFORM(image)
    # ML-003 ResNet-18 expects 3 channels; repeat grayscale channel.
    tensor = tensor.repeat(3, 1, 1)
    return tensor.unsqueeze(0).numpy().astype(np.float32)


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def predict(session, input_name, image_path, idx_to_class):
    input_tensor = preprocess_image(image_path)
    outputs = session.run(None, {input_name: input_tensor})
    probabilities = softmax(outputs[0])[0]
    top_indices = np.argsort(probabilities)[::-1][:3]

    return [
        (
            idx_to_class.get(int(idx), f"class_{int(idx)}"),
            float(probabilities[int(idx)]),
            int(idx),
        )
        for idx in top_indices
    ]


def main():
    print("=" * 70)
    print("MEDICINE WORD INFERENCE - ML-003")
    print("=" * 70)
    print("Segment root :", SEGMENT_ROOT)
    print("Manifest     :", MANIFEST)
    print("ONNX model   :", ONNX_MODEL)
    print()

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}\n"
            "Run the P1 segmentation first."
        )

    idx_to_class = load_class_mapping()
    session = load_session()
    input_name = session.get_inputs()[0].name
    rows = []

    with open(MANIFEST, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            segment_path = PROJECT_ROOT / row["segment_path"]

            if not segment_path.exists():
                print(f"WARNING: missing segment: {segment_path}")
                continue

            predictions = predict(
                session, input_name, segment_path, idx_to_class
            )
            top1, top2, top3 = predictions

            rows.append({
                "region_id": row["region_id"],
                "segment_id": row["segment_id"],
                "ocr_text": row["ocr_text"],
                "segment_path": row["segment_path"],
                "top1_medicine": top1[0],
                "top1_confidence": f"{top1[1]:.6f}",
                "top1_class_id": top1[2],
                "top2_medicine": top2[0],
                "top2_confidence": f"{top2[1]:.6f}",
                "top2_class_id": top2[2],
                "top3_medicine": top3[0],
                "top3_confidence": f"{top3[1]:.6f}",
                "top3_class_id": top3[2],
            })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "region_id", "segment_id", "ocr_text", "segment_path",
        "top1_medicine", "top1_confidence", "top1_class_id",
        "top2_medicine", "top2_confidence", "top2_class_id",
        "top3_medicine", "top3_confidence", "top3_class_id",
    ]

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Segments processed :", len(rows))
    print("Results saved      :", OUTPUT)
    print("=" * 70)


if __name__ == "__main__":
    main()
