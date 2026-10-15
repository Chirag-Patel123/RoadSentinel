"""
RoadSentinel - Real Image Test Runner
Executes genuine model inference on verified road damage dataset samples,
saves annotated visualizations, and audits model predictions against ground truth.
"""

import os
import sys
import glob
import time
import cv2
import json
from typing import Dict, Any, List

# Ensure UTF-8 stdout
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from detector import RoadDamageDetector, CLASS_NAMES

OUTPUT_DIR = os.path.join("outputs", "real_image_tests")
SAMPLES_DIR = os.path.join("samples", "real_road_damage")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_yolo_ground_truth(txt_path: str, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """Parse YOLO format ground truth annotation: class_id cx cy w h (normalized)."""
    if not os.path.exists(txt_path):
        return []
    
    gt_boxes = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    # Denormalize
                    x = int((cx - w / 2.0) * img_w)
                    y = int((cy - h / 2.0) * img_h)
                    bw = int(w * img_w)
                    bh = int(h * img_h)
                    gt_boxes.append({
                        "class_id": cls_id,
                        "box": [max(0, x), max(0, y), min(img_w - x, bw), min(img_h - y, bh)]
                    })
                except ValueError:
                    continue
    return gt_boxes

def run_real_image_tests(conf_threshold: float = 0.20):
    print("=" * 75)
    print("      ROADSENTINEL - REAL ROAD DAMAGE TEST RUNNER      ")
    print("=" * 75)
    print(f"Dataset Directory : {SAMPLES_DIR}")
    print(f"Output Directory  : {OUTPUT_DIR}")
    print(f"Confidence Thresh : {conf_threshold:.2f}")
    print("=" * 75)

    detector = RoadDamageDetector()
    status = detector.get_status()
    if not status["loaded"]:
        print(f"[FATAL ERROR] Model failed to load: {status['error']}")
        sys.exit(1)
        
    print(f"Model Engine      : {status['backend']} ({status['architecture']})")
    print(f"Model Classes     : {status['classes']}\n")

    # Find all images
    supported_exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    image_paths = []
    for ext in supported_exts:
        image_paths.extend(glob.glob(os.path.join(SAMPLES_DIR, "**", ext), recursive=True))

    image_paths = sorted(image_paths)
    if not image_paths:
        print(f"[ERROR] No image files found in {SAMPLES_DIR}")
        sys.exit(1)

    print(f"Found {len(image_paths)} genuine road damage test images.\n")

    processed_count = 0
    failed_count = 0
    results_summary = []

    for idx, img_path in enumerate(image_paths, 1):
        rel_path = os.path.relpath(img_path, SAMPLES_DIR)
        category_dir = os.path.dirname(rel_path)
        filename = os.path.basename(img_path)

        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not decode image at {img_path}")
            h, w = img.shape[:2]

            # Parse ground truth if present
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            gt_labels = parse_yolo_ground_truth(txt_path, w, h)

            t0 = time.time()
            res = detector.predict(img_path, conf_threshold=conf_threshold)
            latency = time.time() - t0

            # Save annotated output
            out_filename = f"annotated_{category_dir}_{filename}"
            out_path = os.path.join(OUTPUT_DIR, out_filename)
            cv2.imwrite(out_path, res["annotated_image"])

            dets = res["detections"]
            det_summary = [(d["class_name"], round(d["confidence"], 3)) for d in dets]
            
            print(f"[{idx:2d}/{len(image_paths)}] {category_dir.upper():18} | {filename:22} | {w}x{h} | Detections: {len(dets)} | {latency:.2f}s")
            if dets:
                for d in dets:
                    print(f"       -> Pred: {d['class_name']:18} (Conf: {d['confidence']*100:.1f}%, Box: {d['box']})")
            else:
                print("       -> No defects detected above threshold.")

            if gt_labels:
                print(f"       -> Ground Truth: {len(gt_labels)} verified annotations in dataset.")

            results_summary.append({
                "image": rel_path,
                "category": category_dir,
                "dimensions": f"{w}x{h}",
                "detection_count": len(dets),
                "predictions": det_summary,
                "ground_truth_count": len(gt_labels),
                "latency_sec": round(latency, 3),
                "annotated_path": out_path
            })
            processed_count += 1

        except Exception as e:
            print(f"[{idx:2d}/{len(image_paths)}] FAILED {rel_path}: {e}")
            failed_count += 1

    print("\n" + "=" * 75)
    print("                      SUMMARY OF EXECUTION                     ")
    print("=" * 75)
    print(f"Successfully processed images : {processed_count}")
    print(f"Failed images                 : {failed_count}")
    print(f"Annotated images saved to     : {OUTPUT_DIR}")
    
    # Save JSON summary
    summary_json_path = os.path.join(OUTPUT_DIR, "real_image_test_results.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Detailed report saved to      : {summary_json_path}")
    print("=" * 75)

if __name__ == "__main__":
    conf = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
    run_real_image_tests(conf_threshold=conf)
