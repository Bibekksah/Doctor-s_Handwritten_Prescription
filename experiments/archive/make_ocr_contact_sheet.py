from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = PROJECT_ROOT / "ml" / "results" / "ocr_ml003" / "crops"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "results" / "ocr_ml003"
OUTPUT_FILE = OUTPUT_DIR / "ocr_regions_contact_sheet.jpg"

COLS = 4
CELL_W = 360
CELL_H = 170
PADDING = 10

def get_font(size=18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()

def main():
    if not CROPS_DIR.exists():
        raise FileNotFoundError(
            f"Crops directory not found:\n{CROPS_DIR}\n\n"
            "Run this first:\n"
            "python -m experiments.test_ocr_ml003_integration"
        )

    crop_files = sorted(
        CROPS_DIR.glob("region_*.jpg"),
        key=lambda p: int(p.stem.split("_")[1])
    )

    if not crop_files:
        raise FileNotFoundError(f"No region_*.jpg files found in {CROPS_DIR}")

    rows = (len(crop_files) + COLS - 1) // COLS
    sheet_w = COLS * CELL_W
    sheet_h = rows * CELL_H

    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    title_font = get_font(18)
    label_font = get_font(16)

    for index, crop_path in enumerate(crop_files):
        row = index // COLS
        col = index % COLS

        x0 = col * CELL_W
        y0 = row * CELL_H

        image = Image.open(crop_path).convert("RGB")

        # Keep aspect ratio while fitting inside the image area.
        max_w = CELL_W - 2 * PADDING
        max_h = CELL_H - 45

        scale = min(max_w / image.width, max_h / image.height)
        new_size = (
            max(1, int(image.width * scale)),
            max(1, int(image.height * scale)),
        )

        image = image.resize(new_size, Image.Resampling.LANCZOS)

        image_x = x0 + (CELL_W - new_size[0]) // 2
        image_y = y0 + 35 + (max_h - new_size[1]) // 2

        draw.rectangle(
            [x0 + 1, y0 + 1, x0 + CELL_W - 2, y0 + CELL_H - 2],
            outline="black",
            width=2,
        )

        region_number = index + 1
        draw.text(
            (x0 + PADDING, y0 + 8),
            f"Region {region_number:02d}",
            fill="black",
            font=title_font,
        )

        sheet.paste(image, (image_x, image_y))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.save(OUTPUT_FILE, quality=95)

    print("=" * 70)
    print("OCR REGION CONTACT SHEET")
    print("=" * 70)
    print(f"Regions found : {len(crop_files)}")
    print(f"Output        : {OUTPUT_FILE}")
    print()
    print("Manual labeling:")
    print("  MEDICINE     = likely medicine-name region")
    print("  NON_MEDICINE = doctor/header/address/instruction/etc.")
    print("  UNCERTAIN    = cannot confidently decide")
    print()
    print("The region numbers match region_XX.jpg.")
    print("=" * 70)

if __name__ == "__main__":
    main()
