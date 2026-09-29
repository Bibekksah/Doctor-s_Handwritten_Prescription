import pandas as pd
from PIL import Image


# Load our standardized metadata
df = pd.read_csv("data/metadata/dataset.csv")


print("=" * 60)
print("IMAGE INSPECTION")
print("=" * 60)


# Inspect first 10 images
for _, row in df.head(10).iterrows():

    image_path = row["image_path"]

    image = Image.open(image_path)

    print("\nImage:", image_path)
    print("Medicine:", row["medicine_name"])
    print("Generic:", row["generic_name"])
    print("Size:", image.size)
    print("Mode:", image.mode)


print("\n" + "=" * 60)
print("IMAGE INSPECTION COMPLETE")
print("=" * 60)