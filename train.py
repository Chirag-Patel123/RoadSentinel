"""
RoadSentinel - Fine-Tuning Pipeline
Reproducible training script for fine-tuning YOLO on custom Road Damage Datasets (RDD).
"""

import os
import sys
import argparse
import json

def train(
    data_yaml: str = "data.yaml",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 8,
    weights: str = "yolov8s.pt",
    project: str = "runs/detect",
    name: str = "road_damage_train"
):
    print("=" * 65)
    print("      ROADSENTINEL - MODEL FINE-TUNING PIPELINE      ")
    print("=" * 65)
    print(f"Dataset Config   : {data_yaml}")
    print(f"Target Epochs    : {epochs}")
    print(f"Image Resolution : {imgsz}x{imgsz}")
    print(f"Batch Size       : {batch_size}")
    print(f"Base Weights     : {weights}")
    print(f"Output Directory : {os.path.join(project, name)}")
    print("=" * 65)

    if not os.path.exists(data_yaml):
        print(f"[ERROR] Dataset configuration file '{data_yaml}' not found.")
        print("To train on a custom road damage dataset:")
        print("  1. Prepare your dataset in YOLO format (images/ & labels/ for train/val)")
        print("  2. Create a data.yaml specifying paths and classes (Pothole, Cracks, etc.)")
        print("  3. Run: python train.py --data path/to/data.yaml --epochs 50")
        sys.exit(1)

    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        print(f"[ENVIRONMENT LIMITATION] Native PyTorch/CUDA training dependencies: {e}")
        print("On systems without AVX2 or CUDA, model inference runs via OpenCV DNN ONNX.")
        print("For model fine-tuning, run on a machine with CUDA GPU (Google Colab / Kaggle / Dedicated Server).")
        sys.exit(1)

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Compute Device   : {device} (CUDA Available: {torch.cuda.is_available()})")

    # Load model checkpoint
    model = YOLO(weights)

    # Train model
    print("\nStarting genuine YOLO fine-tuning pipeline...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        save=True,
        plots=True
    )

    print("\nFine-tuning completed successfully!")
    best_weights = os.path.join(project, name, "weights", "best.pt")
    if os.path.exists(best_weights):
        print(f"Best model saved at: {best_weights}")
        print("Exporting best model to ONNX for CPU-optimized deployment...")
        model.export(format="onnx", imgsz=imgsz)
        print(f"Exported ONNX saved alongside best.pt")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoadSentinel YOLO Road Damage Fine-Tuning")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--weights", type=str, default="yolov8s.pt", help="Initial weights")
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        weights=args.weights
    )
