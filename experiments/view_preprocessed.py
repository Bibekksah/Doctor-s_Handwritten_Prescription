import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image

from CV.preprocessing import get_baseline_transform


df = pd.read_csv(r"data/metadata/dataset.csv")

samples = df[df["split"] == "train"].head(9)

transform = get_baseline_transform()


fig, axes = plt.subplots(
    3,
    3,
    figsize=(12, 5)
)


for ax, (_, row) in zip(
    axes.flat,
    samples.iterrows()
):

    image = Image.open(
        row["image_path"]
    ).convert("RGB")

    processed = transform(image)

    # Tensor [C,H,W] -> [H,W,C]
    processed = processed.permute(
        1, 2, 0
    )

    ax.imshow(processed)

    ax.set_title(
        row["medicine_name"]
    )

    ax.axis("off")


plt.tight_layout()

plt.savefig(
    "experiments/preprocessed_samples.png",
    dpi=150
)

plt.show()