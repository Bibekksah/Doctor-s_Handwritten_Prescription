from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "dataset.csv"
)


def resolve_image_path(image_path):
    """
    Resolve an image path stored in dataset.csv.

    dataset.csv should contain paths relative to PROJECT_ROOT,
    for example:

    data/metadata/Doctor’s Handwritten Prescription BD dataset/
    Training/training_words/0.png
    """

    path = Path(str(image_path))

    # ---------------------------------------------------------
    # 1. Absolute path
    # ---------------------------------------------------------
    if path.is_absolute():
        if path.exists():
            return path

    # ---------------------------------------------------------
    # 2. Correct project-relative path
    # ---------------------------------------------------------
    candidate = PROJECT_ROOT / path

    if candidate.exists():
        return candidate

    # ---------------------------------------------------------
    # 3. Compatibility fallback:
    #    remove duplicated project folder if it exists
    # ---------------------------------------------------------
    parts = path.parts

    for i, part in enumerate(parts):
        if part == "data":
            candidate = PROJECT_ROOT / Path(*parts[i:])

            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"Image not found.\n"
        f"Original path: {image_path}\n"
        f"Project root: {PROJECT_ROOT}\n"
        f"Resolved path attempted: {candidate}"
    )


class PrescriptionDataset(Dataset):

    def __init__(
        self,
        csv_path=CSV_PATH,
        split=None,
        transform=None,
        class_to_idx=None
    ):

        self.data = pd.read_csv(csv_path)

        required_columns = [
            "image_path",
            "medicine_name",
            "generic_name",
            "split"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        if split is not None:
            self.data = self.data[
                self.data["split"] == split
            ].reset_index(drop=True)

        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = resolve_image_path(
            row["image_path"]
        )

        try:
            image = Image.open(image_path).convert("RGB")

        except Exception as error:
            raise RuntimeError(
                f"\nFailed to load image.\n"
                f"Index: {index}\n"
                f"Image path: {image_path}\n"
                f"Error: {error}"
            ) from error

        if self.transform is not None:
            image = self.transform(image)

        # -----------------------------------------------------
        # P1-P5 produce 1-channel images.
        # Model expects 3 channels.
        # -----------------------------------------------------
        if (
            hasattr(image, "ndim")
            and image.ndim == 3
            and image.shape[0] == 1
        ):
            image = image.repeat(3, 1, 1)

        medicine_name = row["medicine_name"]
        generic_name = row["generic_name"]

        if self.class_to_idx is not None:
            label = self.class_to_idx[medicine_name]
        else:
            label = -1

        return (
            image,
            medicine_name,
            generic_name,
            str(image_path),
            label
        )