from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEGMENT_DIR = (
    PROJECT_ROOT
    / "ml/results/ocr_ml003/medicine_word_segments"
)
OUTPUT = (
    PROJECT_ROOT
    / "ml/results/ocr_ml003/medicine_word_segments_contact_sheet.jpg"
)

images = []

for region_dir in sorted(SEGMENT_DIR.glob("region_*")):
    for path in sorted(region_dir.glob("segment_*.png")):
        try:
            image = Image.open(path).convert("RGB")
            images.append((region_dir.name, path.name, image))
        except Exception:
            pass

if not images:
    raise RuntimeError("No segment images found.")

CELL_W = 300
CELL_H = 150
COLS = 4
ROWS = math.ceil(len(images) / COLS)

sheet = Image.new(
    "RGB",
    (COLS * CELL_W, ROWS * CELL_H),
    "white",
)

draw = ImageDraw.Draw(sheet)

for i, (region, segment, image) in enumerate(images):
    row = i // COLS
    col = i % COLS

    x = col * CELL_W
    y = row * CELL_H

    image.thumbnail((CELL_W - 20, CELL_H - 50))

    image_x = x + (CELL_W - image.width) // 2
    image_y = y + 35

    sheet.paste(image, (image_x, image_y))

    draw.text(
        (x + 8, y + 8),
        f"{region} / {segment}",
        fill="black",
    )

sheet.save(OUTPUT, quality=95)

print("=" * 70)
print("MEDICINE WORD SEGMENT CONTACT SHEET")
print("=" * 70)
print("Segments :", len(images))
print("Output   :", OUTPUT)
print("=" * 70)