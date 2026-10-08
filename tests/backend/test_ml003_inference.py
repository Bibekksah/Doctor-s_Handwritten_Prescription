import numpy as np
import pytest

from backend.ml003_inference import ML003InferenceEngine


class FakeSession:
    def __init__(self, output: np.ndarray):
        self.output = output

    def run(self, output_names, inputs):
        return [self.output]


def create_test_engine(output: np.ndarray) -> ML003InferenceEngine:
    engine = object.__new__(ML003InferenceEngine)

    engine.session = FakeSession(output)
    engine.input_name = "input"
    engine.output_name = "output"

    engine.idx_to_class = {
        index: f"Medicine{index}"
        for index in range(78)
    }

    engine._preprocess = lambda image_path: np.zeros(
        (1, 3, 64, 256),
        dtype=np.float32,
    )

    return engine


def test_valid_model_output_returns_prediction():
    logits = np.zeros((1, 78), dtype=np.float32)
    logits[0, 5] = 10.0

    engine = create_test_engine(logits)

    result = engine.predict("test-image.jpg")

    assert result.label == "Medicine5"
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence > 0.9


def test_invalid_output_dimension_is_rejected():
    output = np.zeros((78,), dtype=np.float32)

    engine = create_test_engine(output)

    with pytest.raises(RuntimeError, match="Unexpected model output shape"):
        engine.predict("test-image.jpg")


def test_wrong_class_count_is_rejected():
    output = np.zeros((1, 77), dtype=np.float32)

    engine = create_test_engine(output)

    with pytest.raises(RuntimeError, match="Unexpected model class count"):
        engine.predict("test-image.jpg")


def test_non_finite_logits_are_rejected():
    output = np.zeros((1, 78), dtype=np.float32)
    output[0, 0] = np.nan

    engine = create_test_engine(output)

    with pytest.raises(
        RuntimeError,
        match="Model output contains non-finite values",
    ):
        engine.predict("test-image.jpg")


def test_positive_infinity_logits_are_rejected():
    output = np.zeros((1, 78), dtype=np.float32)
    output[0, 0] = np.inf

    engine = create_test_engine(output)

    with pytest.raises(
        RuntimeError,
        match="Model output contains non-finite values",
    ):
        engine.predict("test-image.jpg")


def test_invalid_batch_size_is_rejected():
    output = np.zeros((2, 78), dtype=np.float32)

    engine = create_test_engine(output)

    with pytest.raises(
        RuntimeError,
        match="Unexpected model batch size",
    ):
        engine.predict("test-image.jpg")


def test_empty_model_output_is_rejected():
    engine = object.__new__(ML003InferenceEngine)

    engine.session = FakeSession(np.zeros((1, 78), dtype=np.float32))
    engine.input_name = "input"
    engine.output_name = "output"
    engine.idx_to_class = {
        index: f"Medicine{index}"
        for index in range(78)
    }

    engine._preprocess = lambda image_path: np.zeros(
        (1, 3, 64, 256),
        dtype=np.float32,
    )

    engine.session.run = lambda output_names, inputs: []

    with pytest.raises(RuntimeError, match="Model returned no output"):
        engine.predict("test-image.jpg")


def test_invalid_medicine_label_is_rejected():
    output = np.zeros((1, 78), dtype=np.float32)
    output[0, 0] = 10.0

    engine = create_test_engine(output)
    engine.idx_to_class[0] = ""

    with pytest.raises(
        RuntimeError,
        match="invalid medicine label",
    ):
        engine.predict("test-image.jpg")