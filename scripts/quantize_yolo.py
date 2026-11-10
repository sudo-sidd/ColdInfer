"""Export YOLO11n at 160px and apply ONNX Runtime INT8 weight quantization."""

from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic
from ultralytics import YOLO


ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "weights" / "yolo11n_int8_160"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    fp32_model = Path(YOLO(str(ROOT / "yolo11n.pt")).export(
        format="onnx",
        imgsz=160,
        simplify=False,
        opset=17,
    ))
    quantized_model = OUTPUT_DIR / "yolo11n_160_dynamic_int8.onnx"
    quantize_dynamic(
        str(fp32_model),
        str(quantized_model),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print(f"FP32_BYTES:{fp32_model.stat().st_size}")
    print(f"INT8_BYTES:{quantized_model.stat().st_size}")
    print(f"OUTPUT:{quantized_model}")


if __name__ == "__main__":
    main()
