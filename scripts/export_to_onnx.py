"""
Export all models from ground-25-pipeline-v3 to ONNX format for web / CDN deployment.
Supports L1 Dewarp, L2 QBox, L3 CBox, and L4 AnswerChoiceNet.
"""

import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

PIPELINE_ROOT = Path(r"E:\playgrounds\ground-25-pipeline-v3")
OUTPUT_DIR = Path("output_onnx")


def export_l1(output_path: Path):
    print("\n--- [1/4] Exporting L1: UVDocNet Dewarp ---")
    sys.path.insert(0, str(PIPELINE_ROOT))
    from models_loader import load_l1_model
    l1_model, _, _ = load_l1_model(str(PIPELINE_ROOT / "l1-dewarp" / "best_model.pkl"))
    l1_model.eval()

    class L1Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            p2d, _ = self.model(x)
            return p2d

    wrapper = L1Wrapper(l1_model)
    wrapper.eval()
    dummy_input = torch.randn(1, 3, 488, 712)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        input_names=["image"],
        output_names=["p2d"],
        opset_version=14,
        dynamo=False
    )
    print(f"Exported L1 to: {output_path} (Size: {output_path.stat().st_size / (1024*1024):.2f} MB)")


def export_l2(output_path: Path):
    print("\n--- [2/4] Exporting L2: YOLO QBox ---")
    model_path = PIPELINE_ROOT / "l2-qbox" / "yolo-qbox.pt"
    yolo = YOLO(str(model_path))
    exported = yolo.export(format="onnx", imgsz=1024, opset=14, dynamic=False)
    # Move to desired destination
    src = Path(exported)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != output_path.resolve():
        if output_path.exists():
            output_path.unlink()
        src.rename(output_path)
    print(f"Exported L2 to: {output_path} (Size: {output_path.stat().st_size / (1024*1024):.2f} MB)")


def export_l3(output_path: Path):
    print("\n--- [3/4] Exporting L3: YOLO CBox ---")
    model_path = PIPELINE_ROOT / "l3-cbox" / "yolo-cbox.pt"
    yolo = YOLO(str(model_path))
    exported = yolo.export(format="onnx", imgsz=1024, opset=14, dynamic=False)
    src = Path(exported)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != output_path.resolve():
        if output_path.exists():
            output_path.unlink()
        src.rename(output_path)
    print(f"Exported L3 to: {output_path} (Size: {output_path.stat().st_size / (1024*1024):.2f} MB)")


def export_l4(output_path: Path):
    print("\n--- [4/4] Exporting L4: AnswerChoiceNet ---")
    sys.path.insert(0, str(PIPELINE_ROOT))
    from models_loader import load_l4_model
    l4_model = load_l4_model(str(PIPELINE_ROOT / "l4-acnet" / "acnet.pt"))
    l4_model.eval()

    class L4Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            return out["choice_logits"], out["state_logits"]

    wrapper = L4Wrapper(l4_model)
    wrapper.eval()
    dummy_input = torch.randn(1, 3, 64, 256)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        input_names=["image"],
        output_names=["choice_logits", "state_logits"],
        dynamic_axes={
            "image": {0: "batch"},
            "choice_logits": {0: "batch"},
            "state_logits": {0: "batch"}
        },
        opset_version=14,
        dynamo=False
    )
    print(f"Exported L4 to: {output_path} (Size: {output_path.stat().st_size / (1024*1024):.2f} MB)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Starting ONNX export from {PIPELINE_ROOT} to {OUTPUT_DIR.resolve()}...")

    export_l1(OUTPUT_DIR / "uvdocnet.onnx")
    export_l2(OUTPUT_DIR / "yolo-qbox.onnx")
    export_l3(OUTPUT_DIR / "yolo-cbox.onnx")
    export_l4(OUTPUT_DIR / "acnet.onnx")

    print("\nAll models exported successfully!")
    print(f"Folder ready to upload to Cloudflare R2 / Hugging Face: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
