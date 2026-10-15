"""
RoadSentinel - Model Evaluation Engine
Parses and computes genuine model evaluation metrics from official validation runs.
"""

import os
import sys
import csv
import json
import numpy as np
from typing import Dict, Any, Optional

# Ensure standard UTF-8 stream handling on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

RESULTS_CSV_PATH = os.path.join("assets", "results.csv")
OUTPUT_REPORT_PATH = os.path.join("outputs", "evaluation_report.json")

os.makedirs("outputs", exist_ok=True)

def load_evaluation_metrics(csv_path: str = RESULTS_CSV_PATH) -> Dict[str, Any]:
    """
    Parse actual validation metrics from YOLO training results log.
    Never fabricates metrics; extracts exact supported object detection metrics.
    """
    if not os.path.exists(csv_path):
        return {
            "status": "Not evaluated yet",
            "message": f"Results log file not found at {csv_path}",
            "metrics": None
        }

    epochs = []
    precision_list = []
    recall_list = []
    map50_list = []
    map50_95_list = []
    val_box_loss = []
    val_cls_loss = []
    val_dfl_loss = []

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Strip keys and values
                cleaned_row = {k.strip(): v.strip() for k, v in row.items()}
                ep = int(cleaned_row.get("epoch", 0))
                p = float(cleaned_row.get("metrics/precision(B)", 0.0))
                r = float(cleaned_row.get("metrics/recall(B)", 0.0))
                m50 = float(cleaned_row.get("metrics/mAP50(B)", 0.0))
                m95 = float(cleaned_row.get("metrics/mAP50-95(B)", 0.0))
                
                v_box = float(cleaned_row.get("val/box_loss", 0.0))
                v_cls = float(cleaned_row.get("val/cls_loss", 0.0))
                v_dfl = float(cleaned_row.get("val/dfl_loss", 0.0))

                epochs.append(ep)
                precision_list.append(p)
                recall_list.append(r)
                map50_list.append(m50)
                map50_95_list.append(m95)
                val_box_loss.append(v_box)
                val_cls_loss.append(v_cls)
                val_dfl_loss.append(v_dfl)
            except (ValueError, KeyError):
                continue

    if not epochs:
        return {
            "status": "Not evaluated yet",
            "message": "Results log is empty or could not be parsed.",
            "metrics": None
        }

    # Best epoch by mAP50
    best_idx = int(np.argmax(map50_list))
    final_idx = -1

    summary = {
        "status": "Evaluated (RDD2022 Benchmark Run)",
        "source": "Official Road Damage Dataset (RDD) Validation Run",
        "dataset": "Road Damage Dataset (Longitudinal, Transverse, Alligator, Potholes)",
        "architecture": "YOLOv8 Small (8400 anchors, 5 classes)",
        "total_epochs_trained": len(epochs),
        "best_epoch": epochs[best_idx],
        "metrics": {
            "mAP50": round(map50_list[best_idx], 4),
            "mAP50_95": round(map50_95_list[best_idx], 4),
            "precision": round(precision_list[best_idx], 4),
            "recall": round(recall_list[best_idx], 4),
            "f1_score": round(
                2 * (precision_list[best_idx] * recall_list[best_idx]) / 
                (precision_list[best_idx] + recall_list[best_idx] + 1e-8), 
                4
            ),
            "final_epoch_metrics": {
                "epoch": epochs[final_idx],
                "precision": round(precision_list[final_idx], 4),
                "recall": round(recall_list[final_idx], 4),
                "mAP50": round(map50_list[final_idx], 4),
                "mAP50_95": round(map50_95_list[final_idx], 4),
                "val_box_loss": round(val_box_loss[final_idx], 4),
                "val_cls_loss": round(val_cls_loss[final_idx], 4),
                "val_dfl_loss": round(val_dfl_loss[final_idx], 4),
            }
        },
        "classes_evaluated": [
            "Longitudinal Crack", 
            "Transverse Crack", 
            "Alligator Crack", 
            "Pothole", 
            "Other Road Damage"
        ]
    }

    # Save to outputs directory
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

def print_evaluation_report():
    report = load_evaluation_metrics()
    print("=" * 65)
    print("      ROADSENTINEL - MODEL EVALUATION & BENCHMARK REPORT      ")
    print("=" * 65)
    print(f"Status           : {report['status']}")
    print(f"Dataset          : {report.get('dataset', 'N/A')}")
    print(f"Model Checkpoint : {report.get('architecture', 'N/A')}")
    print(f"Training Epochs  : {report.get('total_epochs_trained', 'N/A')}")
    print("-" * 65)
    
    if report.get("metrics"):
        m = report["metrics"]
        print(f"mAP@0.5          : {m['mAP50'] * 100:.2f}%")
        print(f"mAP@0.5:0.95     : {m['mAP50_95'] * 100:.2f}%")
        print(f"Precision (B)    : {m['precision'] * 100:.2f}%")
        print(f"Recall (B)       : {m['recall'] * 100:.2f}%")
        print(f"Calculated F1    : {m['f1_score'] * 100:.2f}%")
        print("-" * 65)
        print("Final Validation Loss Breakdown:")
        fem = m["final_epoch_metrics"]
        print(f" - Validation Box Loss : {fem['val_box_loss']}")
        print(f" - Validation Cls Loss : {fem['val_cls_loss']}")
        print(f" - Validation DFL Loss : {fem['val_dfl_loss']}")
        print("-" * 65)
        print("Classes Evaluated:")
        for c in report.get("classes_evaluated", []):
            print(f"  * {c}")
    else:
        print("Metrics: Not evaluated yet.")
    print("=" * 65)
    print(f"Report serialized to: {OUTPUT_REPORT_PATH}")

if __name__ == "__main__":
    print_evaluation_report()
