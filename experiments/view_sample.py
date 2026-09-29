import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


df = pd.read_csv("data/metadata/dataset.csv")

# Take 9 training samples
samples = df[df["split"] == "train"].head(9)


fig, axes = plt.subplots(3, 3, figsize=(10, 8))

for ax, (_, row) in zip(axes.flat, samples.iterrows()):

    image = Image.open(row["image_path"])

    ax.imshow(image)
    ax.set_title(
        f"{row['medicine_name']} → {row['generic_name']}"
    )
    ax.axis("off")


plt.tight_layout()

plt.savefig(
    "experiments/sample_images.png",
    dpi=150
)

plt.show()