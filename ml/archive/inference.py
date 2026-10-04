from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort

from PIL import Image

from CV.preprocessing import get_grayscale_transform
from ml.dataloader import create_class_mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "optimization"
    / "ML003_resnet18.onnx"
)


class MedicineRecognizer:
    """
    ML-003 ONNX inference engine.

    Input:
        Handwritten medicine-word image

    Output:
        Medicine predictions with confidence scores
    """

    def __init__(self, model_path=None):

        if model_path is None:
            model_path = MODEL_PATH

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        # Class mapping used during training
        self.class_to_idx, self.idx_to_class = create_class_mapping()

        # Same preprocessing used by ML-003
        self.transform = get_grayscale_transform()

        # CPU deployment
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image):
        """
        Convert PIL image into ML-003 input tensor.
        """

        if not isinstance(image, Image.Image):
            image = Image.open(image)

        image = image.convert("RGB")

        tensor = self.transform(image)

        # ML-003 ResNet expects 3 channels.
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)

        tensor = tensor.unsqueeze(0)

        return tensor.numpy().astype(np.float32)

    def predict(self, image, top_k=3):
        """
        Predict medicine name.

        Returns:
            Top-k medicine predictions with confidence.
        """

        input_tensor = self.preprocess(image)

        outputs = self.session.run(
            [self.output_name],
            {
                self.input_name: input_tensor
            },
        )

        logits = outputs[0][0]

        # Stable softmax
        logits = logits - np.max(logits)

        probabilities = np.exp(logits)
        probabilities = probabilities / np.sum(probabilities)

        # Highest probabilities first
        indices = np.argsort(probabilities)[::-1][:top_k]

        predictions = []

        for index in indices:

            medicine_name = self.idx_to_class[int(index)]

            confidence = float(probabilities[index])

            predictions.append(
                {
                    "medicine_name": medicine_name,
                    "confidence": confidence,
                    "confidence_percent": round(
                        confidence * 100,
                        2,
                    ),
                }
            )

        return predictions

    def predict_top1(self, image):
        """
        Return only the highest-confidence prediction.
        """

        predictions = self.predict(
            image,
            top_k=1,
        )

        return predictions[0]