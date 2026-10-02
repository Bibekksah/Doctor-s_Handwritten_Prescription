import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from CV.preprocessing import get_baseline_transform
from PIL import Image


def main():
    print("=" * 60)
    print("TEST PREPROCESSING PIPELINE")
    print("=" * 60)

    transform = get_baseline_transform()
    dummy_img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    transformed = transform(dummy_img)

    print(f"Input Size: {dummy_img.size}")
    print(f"Output Tensor Shape: {transformed.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()