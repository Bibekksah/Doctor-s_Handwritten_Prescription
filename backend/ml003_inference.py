from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from backend.inference import InferenceResult, ModelInfo
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


class ML003InferenceEngine:
    """
    Production inference engine for the ML003 ResNet-18 ONNX model.

    The preprocessing and class mapping must remain identical to the
    preprocessing and class mapping used during ML003 training/evaluation.
    """

    VERSION = "ML003-resnet18-onnx-1.0.0"
    FRAMEWORK = "onnxruntime"
    EXPECTED_CLASS_COUNT = 78

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = Path(model_path or MODEL_PATH)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ML003 model not found: {self.model_path}"
            )

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if len(inputs) != 1:
            raise RuntimeError(
                f"Expected exactly one model input, found {len(inputs)}."
            )

        if len(outputs) != 1:
            raise RuntimeError(
                f"Expected exactly one model output, found {len(outputs)}."
            )

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

        self.transform = get_grayscale_transform()

        # Verify the production mapping against the dataset mapping.
        _, self.idx_to_class = create_class_mapping()

        if len(self.idx_to_class) != self.EXPECTED_CLASS_COUNT:
            raise RuntimeError(
                f"Expected {self.EXPECTED_CLASS_COUNT} medicine classes, "
                f"found {len(self.idx_to_class)}."
            )

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            version=self.VERSION,
            framework=self.FRAMEWORK,
            model_path=str(self.model_path),
        )

    def _preprocess(self, image_path: Path) -> np.ndarray:
        """
        Convert the uploaded image into the exact input representation
        expected by ML003.
        """
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)

        # ML003 was trained with grayscale input duplicated to
        # three channels for ResNet-18.
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)

        tensor = tensor.unsqueeze(0)

        return tensor.numpy().astype(np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """
        Numerically stable softmax.

        The caller is responsible for validating that logits are finite.
        """
        logits = logits - np.max(logits)

        probabilities = np.exp(logits)
        probability_sum = np.sum(probabilities)

        if not np.isfinite(probability_sum) or probability_sum <= 0:
            raise RuntimeError(
                "Model output produced invalid probabilities."
            )

        probabilities /= probability_sum

        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError(
                "Model output produced non-finite probabilities."
            )

        return probabilities

    def _validate_output(
        self,
        output: object,
    ) -> np.ndarray:
        """
        Validate and extract the single-image logits returned by ONNX.

        Expected ONNX output shape:

            [1, 78]

        Returns:

            [78] logits
        """
        if not isinstance(output, np.ndarray):
            raise RuntimeError(
                "Model output is not a NumPy array."
            )

        if output.ndim != 2:
            raise RuntimeError(
                "Unexpected model output shape. "
                f"Expected 2 dimensions, received shape {output.shape}."
            )

        if output.shape[0] != 1:
            raise RuntimeError(
                "Unexpected model batch size. "
                f"Expected 1, received {output.shape[0]}."
            )

        if output.shape[1] != self.EXPECTED_CLASS_COUNT:
            raise RuntimeError(
                "Unexpected model class count. "
                f"Expected {self.EXPECTED_CLASS_COUNT}, "
                f"received {output.shape[1]}."
            )

        logits = output[0]

        if not np.all(np.isfinite(logits)):
            raise RuntimeError(
                "Model output contains non-finite values."
            )

        return logits

    def predict(self, image_path: Path) -> InferenceResult:
        """
        Run ML003 inference for a single image.

        The model output is validated before converting it into a
        medicine prediction.
        """
        input_tensor = self._preprocess(image_path)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor},
        )

        if not outputs:
            raise RuntimeError(
                "Model returned no output."
            )

        logits = self._validate_output(outputs[0])

        probabilities = self._softmax(logits)

        predicted_index = int(np.argmax(probabilities))

        if not 0 <= predicted_index < len(self.idx_to_class):
            raise RuntimeError(
                "Model returned an invalid prediction index."
            )

        try:
            label = self.idx_to_class[predicted_index]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Model prediction could not be mapped to a medicine."
            ) from exc

        if not isinstance(label, str) or not label.strip():
            raise RuntimeError(
                "Model prediction produced an invalid medicine label."
            )

        confidence = float(probabilities[predicted_index])

        if not np.isfinite(confidence):
            raise RuntimeError(
                "Model prediction produced a non-finite confidence."
            )

        if not 0.0 <= confidence <= 1.0:
            raise RuntimeError(
                "Model prediction produced an invalid confidence."
            )

        return InferenceResult(
            label=label,
            confidence=confidence,
        )