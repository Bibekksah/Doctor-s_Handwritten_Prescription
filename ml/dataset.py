import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class PrescriptionDataset(Dataset):

    def __init__(
        self,
        csv_path,
        split=None,
        transform=None
    ):

        self.data = pd.read_csv(csv_path)
        self.transform = transform

        if split is not None:
            self.data = self.data[
                self.data["split"] == split
            ].reset_index(drop=True)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = Path(row["image_path"])

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "medicine_name": row["medicine_name"],
            "generic_name": row["generic_name"],
            "image_path": str(image_path)
        }