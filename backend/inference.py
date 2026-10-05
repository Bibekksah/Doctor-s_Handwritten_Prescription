from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ModelInfo:
    version: str
    framework: str
    model_path: str | None = None


@dataclass(frozen=True)
class InferenceResult:
    label: str
    confidence: float


class InferenceEngine(Protocol):
    @property
    def model_info(self) -> ModelInfo:
        ...

    def predict(self, image_path: Path) -> InferenceResult:
        ...


class MockInferenceEngine:
    """
    Temporary inference implementation.

    This simulates the ML model until Member 3's
    trained model is ready.
    """

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            version="mock-0.1.0",
            framework="mock",
            model_path=None,
        )

    def predict(self, image_path: Path) -> InferenceResult:
        return InferenceResult(
            label="Aceta",
            confidence=0.94,
        )