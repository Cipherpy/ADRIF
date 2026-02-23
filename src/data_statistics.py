from pathlib import Path

# Change this to your dataset root directory
dataset_root = Path("/home/reshma/ADRIF/ADRIF/model_data_old")

# Image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP"}

def count_images(folder):
    return len([f for f in folder.rglob("*") if f.suffix in IMG_EXTS])

for split in ["train", "test", "val"]:
    folder = dataset_root / split
    if folder.exists():
        count = count_images(folder)
        print(f"{split.upper()} images: {count}")
    else:
        print(f"{split.upper()} folder not found")
