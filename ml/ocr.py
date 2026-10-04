from pathlib import Path
from typing import List, Dict, Union

import cv2
import numpy as np
from PIL import Image


class PrescriptionOCR:
    """
    OCR preprocessing and handwritten text-region extraction module.

    Current responsibility:
    1. Load prescription image
    2. Convert to grayscale
    3. Reduce noise
    4. Detect likely handwritten/text regions
    5. Crop detected regions

    Note:
    This module does not claim to perform reliable medical-text OCR.
    Cropped regions can be passed to the ML-003 medicine classifier.
    """

    def __init__(
        self,
        min_width: int = 30,
        min_height: int = 10,
        min_area: int = 200,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.min_area = min_area

    # ---------------------------------------------------------
    # IMAGE LOADING
    # ---------------------------------------------------------

    def load_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Load an image and return it as an OpenCV BGR image.
        """

        if isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        image_path = Path(image)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Prescription image not found: {image_path}"
            )

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        return image

    # ---------------------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate preprocessing stages used for text-region detection.
        """

        if image is None:
            raise ValueError("Input image is None.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Reduce small noise while preserving handwriting
        denoised = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # Adaptive threshold works better than a fixed threshold
        # when the prescription has uneven lighting.
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            4,
        )

        # Connect nearby characters into possible word regions
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (15, 3)
        )

        connected = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel
        )

        return {
            "gray": gray,
            "denoised": denoised,
            "binary": binary,
            "connected": connected,
        }

    # ---------------------------------------------------------
    # TEXT REGION DETECTION
    # ---------------------------------------------------------

    def detect_regions(
        self,
        image: np.ndarray,
    ) -> List[Dict]:
        """
        Detect candidate handwritten/text regions.

        Returns bounding boxes sorted from top to bottom.
        """

        stages = self.preprocess(image)

        connected = stages["connected"]

        contours, _ = cv2.findContours(
            connected,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        regions = []

        image_height, image_width = image.shape[:2]

        for contour in contours:

            x, y, width, height = cv2.boundingRect(contour)

            area = width * height

            if width < self.min_width:
                continue

            if height < self.min_height:
                continue

            if area < self.min_area:
                continue

            # Ignore regions covering almost the entire page.
            if width > image_width * 0.95:
                continue

            if height > image_height * 0.50:
                continue

            regions.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "width": int(width),
                    "height": int(height),
                    "area": int(area),
                }
            )

        # Sort approximately reading-order:
        # top → bottom, then left → right.
        regions.sort(
            key=lambda r: (
                r["y"],
                r["x"],
            )
        )

        return regions

    # ---------------------------------------------------------
    # CROP REGIONS
    # ---------------------------------------------------------

    def crop_regions(
        self,
        image: np.ndarray,
        regions: List[Dict],
        padding: int = 8,
    ) -> List[Dict]:
        """
        Crop detected regions from the original image.
        """

        image_height, image_width = image.shape[:2]

        crops = []

        for index, region in enumerate(regions):

            x = region["x"]
            y = region["y"]
            width = region["width"]
            height = region["height"]

            x1 = max(0, x - padding)
            y1 = max(0, y - padding)

            x2 = min(
                image_width,
                x + width + padding
            )

            y2 = min(
                image_height,
                y + height + padding
            )

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crops.append(
                {
                    "region_id": index,
                    "bbox": {
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1,
                    },
                    "image": crop,
                }
            )

        return crops

    # ---------------------------------------------------------
    # COMPLETE REGION PIPELINE
    # ---------------------------------------------------------

    def extract_text_regions(
        self,
        image: Union[str, Path, Image.Image],
        padding: int = 8,
    ) -> List[Dict]:
        """
        Complete pipeline:

        image
          ↓
        preprocessing
          ↓
        region detection
          ↓
        cropping
        """

        image = self.load_image(image)

        regions = self.detect_regions(image)

        crops = self.crop_regions(
            image,
            regions,
            padding=padding,
        )

        return crops

    def draw_regions(self, image, regions):
        """
        Draw detected regions on an image.

        Accepts:
        - image path
        - PIL Image
        - OpenCV numpy array
        """

        if isinstance(image, np.ndarray):
            output = image.copy()
        else:
            output = self.load_image(image)

        for index, region in enumerate(regions):

            x = region["x"]
            y = region["y"]
            width = region["width"]
            height = region["height"]

            cv2.rectangle(
                output,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                output,
                f"R{index}",
                (x, max(20, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return output

    # ---------------------------------------------------------
    # DEBUG / VISUALIZATION IMAGE
    # ---------------------------------------------------------

def draw_regions(
    self,
    image,
    regions,
):
    """
    Draw detected regions on an image.

    Accepts either:
    - image path
    - PIL Image
    - OpenCV numpy array
    """

    if isinstance(image, np.ndarray):
        output = image.copy()
    else:
        output = self.load_image(image)

    for index, region in enumerate(regions):

        x = region["x"]
        y = region["y"]
        width = region["width"]
        height = region["height"]

        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            output,
            f"R{index}",
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return output