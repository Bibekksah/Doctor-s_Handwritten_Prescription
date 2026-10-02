import pandas as pd
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PrescriptionDataset(Dataset):

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

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = PROJECT_ROOT / row["image_path"]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        medicine_name = row["medicine_name"]

        result = {
            "image": image,
            "medicine_name": medicine_name,
            "generic_name": row["generic_name"],
            "image_path": str(image_path)
        }

        if self.class_to_idx is not None:

            result["label"] = self.class_to_idx[
                medicine_name
            ]

        return result