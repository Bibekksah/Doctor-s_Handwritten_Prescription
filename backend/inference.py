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
