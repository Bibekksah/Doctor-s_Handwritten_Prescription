import sys
from pathlib import Path

from PIL import Image
import pandas as pd
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_image_path(raw_path: str) -> Path:
    """Resolves CSV image path entries into valid system Path objects,

    stripping duplicate root directory prefixes across OS environments.
    """
    raw_str = str(raw_path).strip()

    if "Doctor-s_Handwritten_Prescription" in raw_str:
        clean_relative = raw_str.split("Doctor-s_Handwritten_Prescription")[
            -1
        ].lstrip("/\\")
    elif "data/metadata" in raw_str:
        clean_relative = raw_str.split("data/metadata")[-1].lstrip("/\\")
        return PROJECT_ROOT / "data" / "metadata" / clean_relative
    else:
        clean_relative = raw_str.lstrip("/\\")

    return PROJECT_ROOT / Path(clean_relative)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PrescriptionDataset(Dataset):

<<<<<<< HEAD
    def __init__(
        self,
        csv_path,
        split=None,
        transform=None,
        class_to_idx=None
    ):

        self.data = pd.read_csv(csv_path)

        self.transform = transform

        self.class_to_idx = class_to_idx

        if split is not None:

            self.data = self.data[
                self.data["split"] == split
            ].reset_index(drop=True)

=======
    def __init__(self, csv_file, split="train", transform=None):
        self.df = pd.read_csv(csv_file)
        if "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.transform = transform

>>>>>>> origin/computervision
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = resolve_image_path(row["image_path"])

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at path: {image_path}")

<<<<<<< HEAD
        row = self.data.iloc[index]

        image_path = PROJECT_ROOT / row["image_path"]

=======
>>>>>>> origin/computervision
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        medicine_name = row["medicine_name"]

        result = {
            "image": image,
<<<<<<< HEAD
            "medicine_name": medicine_name,
            "generic_name": row["generic_name"],
            "image_path": str(image_path)
        }

        if self.class_to_idx is not None:

            result["label"] = self.class_to_idx[
                medicine_name
            ]

        return result
=======
            "medicine_name": row.get("medicine_name", ""),
            "generic_name": row.get("generic_name", ""),
            "image_path": str(image_path),
        }
>>>>>>> origin/computervision
