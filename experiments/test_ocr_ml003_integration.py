"""
OCR + ML-003 Integration Experiment

Pipeline:
    Full prescription image
        -> existing PaddleOCR result JSON
        -> detected regions
        -> crop each region
        -> P1 grayscale preprocessing
        -> ML-003 ONNX inference
        -> Top-3 medicine predictions

IMPORTANT:
- This is an integration experiment, NOT a full-prescription accuracy benchmark.
- ML-003 was trained on individual handwritten medicine-name images.
- The existing OCR result is used only to obtain candidate regions.
- The ML-003 checkpoint/model is not modified.
"""

from pathlib import Path
import json
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

from CV.preprocessing import get_grayscale_transform


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "ocr_samples"
    / "images.jpeg"
)

OCR_RESULT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "ocr"
    / "existing_ocr_result.json"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "optimization"
    / "ML003_resnet18.onnx"
)

CLASS_MAPPING_PATH = (
    PROJECT_ROOT
    / "ml"
    / "config"
    / "class_mapping.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "ocr_ml003"
)

CROPS_DIR = OUTPUT_DIR / "crops"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

RESULT_JSON = OUTPUT_DIR / "OCR_ML003_integration.json"
VISUALIZATION_PATH = OUTPUT_DIR / "ocr_ml003_predictions.jpg"

TOP_K = 3
PADDING_RATIO = 0.05


# ============================================================
# CLASS MAPPING
# ============================================================

def load_class_names():
    """
    Load medicine class mapping.

    Supports:
        {"0": "Aceta", "1": "Ace", ...}

    and common nested formats.
    """

    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):

        # Format:
        # {"0": "Aceta", "1": "Ace", ...}
        if all(str(k).isdigit() for k in data.keys()):
            return [
                data[str(i)]
                for i in range(len(data))
            ]

        # Possible nested formats
        for key in (
            "idx_to_class",
            "index_to_class",
            "classes",
            "class_names",
        ):

            value = data.get(key)

            if isinstance(value, dict):
                return [
                    value[str(i)]
                    for i in range(len(value))
                ]

            if isinstance(value, list):
                return value

    if isinstance(data, list):
        return data

    raise ValueError(
        f"Unsupported class mapping format: "
        f"{CLASS_MAPPING_PATH}"
    )


# ============================================================
# LOAD OCR RESULTS
# ============================================================

def load_ocr_regions():

    with open(
        OCR_RESULT_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    regions = data.get("regions")

    if regions is None:
        regions = data.get("results")

    if regions is None:
        regions = data.get("ocr_results")

    if not isinstance(regions, list):

        raise ValueError(
            "Could not find OCR region list.\n"
            f"Available JSON keys: {list(data.keys())}"
        )

    return regions


# ============================================================
# OCR REGION PARSER
# ============================================================

def get_region_values(region):

    text = region.get("text", "")

    confidence = region.get(
        "recognition_confidence"
    )

    if confidence is None:
        confidence = region.get("confidence")

    if confidence is None:
        confidence = region.get("score")

    bbox = region.get("bounding_box")

    if bbox is None:
        bbox = region.get("bbox")

    # Format:
    # {"x": ..., "y": ..., "width": ..., "height": ...}

    if isinstance(bbox, dict):

        x = bbox.get("x")
        y = bbox.get("y")
        w = bbox.get("width")
        h = bbox.get("height")

    # Format:
    # [x, y, width, height]

    elif isinstance(bbox, (list, tuple)):

        if len(bbox) != 4:
            raise ValueError(
                f"Invalid bbox: {bbox}"
            )

        x, y, w, h = bbox

    else:

        raise ValueError(
            f"Unsupported bbox format: {bbox}"
        )

    if None in (x, y, w, h):

        raise ValueError(
            f"Incomplete bbox: {bbox}"
        )

    return (
        str(text),
        float(confidence)
        if confidence is not None
        else None,
        float(x),
        float(y),
        float(w),
        float(h),
    )


# ============================================================
# CROP REGION
# ============================================================

def crop_with_padding(
    image,
    x,
    y,
    w,
    h,
    padding_ratio=PADDING_RATIO,
):

    image_width, image_height = image.size

    pad_x = w * padding_ratio
    pad_y = h * padding_ratio

    left = max(
        0,
        int(round(x - pad_x))
    )

    top = max(
        0,
        int(round(y - pad_y))
    )

    right = min(
        image_width,
        int(round(x + w + pad_x))
    )

    bottom = min(
        image_height,
        int(round(y + h + pad_y))
    )

    if right <= left or bottom <= top:

        raise ValueError(
            f"Invalid crop: "
            f"{left}, {top}, {right}, {bottom}"
        )

    crop = image.crop(
        (
            left,
            top,
            right,
            bottom,
        )
    )

    return crop, (
        left,
        top,
        right,
        bottom,
    )


# ============================================================
# SOFTMAX
# ============================================================

def softmax(logits):

    logits = logits.astype(
        np.float32
    )

    logits = (
        logits
        - np.max(logits)
    )

    exp_values = np.exp(logits)

    return (
        exp_values
        / np.sum(exp_values)
    )


# ============================================================
# ML-003 INFERENCE
# ============================================================

def run_ml003(
    session,
    transform,
    class_names,
    crop,
):

    # P1 grayscale preprocessing
    tensor = transform(crop)

    # P1 produces:
    # [1, 64, 256]

    # ML-003 ResNet-18 expects:
    # [3, 64, 256]

    if (
        tensor.ndim == 3
        and tensor.shape[0] == 1
    ):

        tensor = tensor.repeat(
            3,
            1,
            1,
        )

    input_tensor = (
        tensor
        .unsqueeze(0)
        .numpy()
        .astype(np.float32)
    )

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    start = time.perf_counter()

    outputs = session.run(
        None,
        {
            input_name: input_tensor
        },
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    logits = outputs[0][0]

    probabilities = softmax(
        logits
    )

    top_indices = np.argsort(
        probabilities
    )[::-1][:TOP_K]

    predictions = []

    for rank, index in enumerate(
        top_indices,
        start=1,
    ):

        index = int(index)

        if index < len(class_names):
            medicine_name = (
                class_names[index]
            )
        else:
            medicine_name = (
                f"class_{index}"
            )

        predictions.append(
            {
                "rank": rank,
                "class_index": index,
                "medicine_name": medicine_name,
                "confidence": float(
                    probabilities[index]
                ),
            }
        )

    return (
        predictions,
        latency_ms,
    )


# ============================================================
# VISUALIZATION
# ============================================================

def draw_results(
    image,
    results,
):

    output = image.copy().convert(
        "RGB"
    )

    draw = ImageDraw.Draw(
        output
    )

    for result in results:

        left, top, right, bottom = (
            result["crop_box"]
        )

        top1 = (
            result[
                "ml003_predictions"
            ][0]
        )

        draw.rectangle(
            (
                left,
                top,
                right,
                bottom,
            ),
            outline="red",
            width=2,
        )

        label = (
            f"#{result['region_id']} "
            f"{top1['medicine_name']} "
            f"{top1['confidence']:.2f}"
        )

        draw.text(
            (
                left,
                max(0, top - 14),
            ),
            label,
            fill="red",
        )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("OCR + ML-003 INTEGRATION EXPERIMENT")
    print("=" * 75)

    print(
        f"Input image       : {IMAGE_PATH}"
    )

    print(
        f"OCR result        : {OCR_RESULT_PATH}"
    )

    print(
        f"ML-003 ONNX       : {MODEL_PATH}"
    )

    print(
        f"Class mapping     : {CLASS_MAPPING_PATH}"
    )

    print()

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    required_files = [
        IMAGE_PATH,
        OCR_RESULT_PATH,
        MODEL_PATH,
        CLASS_MAPPING_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    regions = load_ocr_regions()

    class_names = load_class_names()

    print(
        f"Image size        : "
        f"{image.width} x {image.height}"
    )

    print(
        f"OCR regions       : "
        f"{len(regions)}"
    )

    print(
        f"ML-003 classes    : "
        f"{len(class_names)}"
    )

    print()

    # --------------------------------------------------------
    # LOAD ONNX MODEL
    # --------------------------------------------------------

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=[
            "CPUExecutionProvider"
        ],
    )

    print(
        "ONNX provider     : "
        f"{session.get_providers()[0]}"
    )

    print(
        "Model input       : "
        f"{session.get_inputs()[0].shape}"
    )

    print(
        "Model output      : "
        f"{session.get_outputs()[0].shape}"
    )

    print()

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    transform = (
        get_grayscale_transform()
    )

    all_results = []

    total_latency = 0.0
    successful = 0

    print(
        "Running ML-003 on "
        "every OCR region..."
    )

    print()

    # --------------------------------------------------------
    # PROCESS EVERY OCR REGION
    # --------------------------------------------------------

    for region_id, region in enumerate(
        regions,
        start=1,
    ):

        try:

            (
                text,
                ocr_confidence,
                x,
                y,
                w,
                h,
            ) = get_region_values(
                region
            )

            crop, crop_box = (
                crop_with_padding(
                    image,
                    x,
                    y,
                    w,
                    h,
                )
            )

            crop_path = (
                CROPS_DIR
                / f"region_{region_id:02d}.jpg"
            )

            crop.save(
                crop_path,
                quality=95,
            )

            predictions, latency_ms = (
                run_ml003(
                    session,
                    transform,
                    class_names,
                    crop,
                )
            )

            total_latency += (
                latency_ms
            )

            successful += 1

            result = {

                "region_id":
                    region_id,

                "ocr_text":
                    text,

                "ocr_confidence":
                    ocr_confidence,

                "ocr_bbox": {

                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                },

                "crop_box":
                    list(crop_box),

                "crop_path":
                    str(
                        crop_path
                        .relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "ml003_predictions":
                    predictions,

                "ml003_top1_medicine":
                    predictions[0][
                        "medicine_name"
                    ],

                "ml003_top1_confidence":
                    predictions[0][
                        "confidence"
                    ],

                "ml003_latency_ms":
                    latency_ms,
            }

            all_results.append(
                result
            )

            # ------------------------------------------------
            # PRINT RESULT
            # ------------------------------------------------

            print(
                f"Region {region_id:02d}"
            )

            print(
                f"  OCR text       : "
                f"{text}"
            )

            if (
                ocr_confidence
                is not None
            ):

                print(
                    f"  OCR confidence : "
                    f"{ocr_confidence:.4f}"
                )

            print(
                "  ML-003 Top-3:"
            )

            for prediction in (
                predictions
            ):

                print(
                    f"    "
                    f"{prediction['rank']}. "
                    f"{prediction['medicine_name']:<18} "
                    f"{prediction['confidence']:.4f}"
                )

            print(
                f"  ML-003 latency : "
                f"{latency_ms:.3f} ms"
            )

            print(
                "-" * 75
            )

        except Exception as exc:

            print(
                f"Region {region_id:02d} "
                f"FAILED: {exc}"
            )

            all_results.append(
                {
                    "region_id":
                        region_id,
                    "error":
                        str(exc),
                }
            )

    # --------------------------------------------------------
    # SAVE VISUALIZATION
    # --------------------------------------------------------

    valid_results = [
        result
        for result in all_results
        if "crop_box" in result
    ]

    visualization = (
        draw_results(
            image,
            valid_results,
        )
    )

    visualization.save(
        VISUALIZATION_PATH,
        quality=95,
    )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    output = {

        "experiment":
            "OCR_ML003_INTEGRATION",

        "description":
            (
                "Qualitative integration test "
                "of PaddleOCR candidate regions "
                "with the ML-003 ResNet-18 "
                "ONNX medicine classifier."
            ),

        "input_image":
            str(
                IMAGE_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),

        "ocr_result_file":
            str(
                OCR_RESULT_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),

        "model":
            str(
                MODEL_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),

        "image_size": {

            "width":
                image.width,

            "height":
                image.height,
        },

        "ocr_region_count":
            len(regions),

        "successful_regions":
            successful,

        "failed_regions":
            len(regions) - successful,

        "ml003_class_count":
            len(class_names),

        "preprocessing":
            "P1 grayscale",

        "top_k":
            TOP_K,

        "padding_ratio":
            PADDING_RATIO,

        "inference_provider":
            "CPUExecutionProvider",

        "total_ml003_latency_ms":
            total_latency,

        "mean_ml003_latency_ms":
            (
                total_latency
                / successful
                if successful
                else None
            ),

        "results":
            all_results,

        "visualization":
            str(
                VISUALIZATION_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
    }

    with open(
        RESULT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()

    print("=" * 75)
    print(
        "OCR + ML-003 INTEGRATION COMPLETE"
    )
    print("=" * 75)

    print(
        f"OCR regions          : "
        f"{len(regions)}"
    )

    print(
        f"Successful regions   : "
        f"{successful}"
    )

    print(
        f"Failed regions       : "
        f"{len(regions) - successful}"
    )

    if successful:

        print(
            f"Mean ML-003 latency  : "
            f"{total_latency / successful:.3f} ms"
        )

    print(
        f"Visualization        : "
        f"{VISUALIZATION_PATH}"
    )

    print(
        f"JSON report          : "
        f"{RESULT_JSON}"
    )

    print(
        f"Crops                : "
        f"{CROPS_DIR}"
    )

    print()

    print("IMPORTANT:")
    print(
        "This is an integration experiment, "
        "not a full-prescription accuracy benchmark."
    )

    print(
        "ML-003 was trained on individual "
        "handwritten medicine-name images."
    )

    print("=" * 75)


if __name__ == "__main__":
    main()