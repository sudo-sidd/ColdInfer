"""Run the downloaded COCO8 validation images through the ColdInfer YOLO target."""

from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.engine import run_yolo_simulation


IMAGE_DIR = PROJECT_ROOT / "data" / "yolo-coco8" / "images" / "val"


def main() -> None:
    images = sorted(IMAGE_DIR.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No COCO8 images found in {IMAGE_DIR}")

    total = 0
    for image_path in images:
        result = run_yolo_simulation(Image.open(image_path).convert("RGB"), arena_kb=520)
        count = len(result["detections"])
        total += count
        print(f"{image_path.name}: {count} detections")
    print(f"SMOKE_IMAGES:{len(images)}")
    print(f"SMOKE_DETECTIONS:{total}")


if __name__ == "__main__":
    main()
