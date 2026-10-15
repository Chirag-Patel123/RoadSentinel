import cv2
import numpy as np
import time
import os
from detector import RoadDamageDetector, CLASS_NAMES

print("=== PHASE 3: REAL MODEL INFERENCE AUDIT ===")

model_path = "models/road_damage_yolov8.onnx"
assert os.path.exists(model_path), f"Model file missing at: {model_path}"
print(f"Model path: {model_path} ({os.path.getsize(model_path):,} bytes)")

detector = RoadDamageDetector(model_path)
status = detector.get_status()
print(f"Detector Status: loaded={status['loaded']}, backend={status['backend']}, classes={len(status['classes'])}")

test_images = [
    "samples/featured/alligator_crack_sample.jpg",
    "samples/featured/longitudinal_crack_sample.jpg",
    "samples/featured/transverse_crack_sample.jpg",
    "samples/sample_road_4.jpg"
]

results_table = []

for img_path in test_images:
    print(f"\n--- Testing Image: {img_path} ---")
    assert os.path.exists(img_path), f"Image not found: {img_path}"
    img = cv2.imread(img_path)
    assert img is not None, f"Failed to load image: {img_path}"
    h, w, c = img.shape
    print(f"Original dimensions: {w}x{h}, Channels: {c}")
    
    t0 = time.time()
    res = detector.predict(img_path, conf_threshold=0.20, nms_threshold=0.45)
    elapsed = time.time() - t0
    
    print(f"Inference completed in: {elapsed:.3f}s (Engine reported: {res['inference_time_sec']:.3f}s)")
    print(f"Total detections: {res['total_detections']}")
    print(f"Overall severity: {res['overall_severity']}")
    print(f"Class breakdown: {res['class_counts']}")
    
    dets = res["detections"]
    det_classes = [d["class_name"] for d in dets]
    det_confs = [round(d["confidence"], 4) for d in dets]
    
    results_table.append({
        "image": img_path,
        "count": res["total_detections"],
        "classes": det_classes,
        "confs": det_confs,
        "time_sec": round(elapsed, 3)
    })
    
    # Check bounding box validity and confidence range
    for idx, d in enumerate(dets):
        x, y, bw, bh = d["box"]
        conf = d["confidence"]
        cls_name = d["class_name"]
        assert 0.0 <= conf <= 1.0, f"Invalid confidence score: {conf}"
        assert not np.isnan(conf) and not np.isinf(conf), f"Non-finite confidence: {conf}"
        assert x >= 0 and y >= 0, f"Negative coordinates: ({x}, {y})"
        assert x + bw <= w and y + bh <= h, f"Box exceeds boundaries: ({x}+{bw}>{w} or {y}+{bh}>{h})"
        assert cls_name in CLASS_NAMES, f"Unknown class name: {cls_name}"
        print(f"  Detection #{idx+1}: {cls_name} | Conf: {conf*100:.2f}% | Severity: {d['severity']} | Box: ({x}, {y}, {bw}, {bh})")
    
    # Check annotated image
    annot = res["annotated_image"]
    assert annot is not None, "Annotated image is None"
    assert annot.shape == (h, w, c), f"Annotated shape mismatch: {annot.shape} vs {(h, w, c)}"
    assert annot.dtype == np.uint8, "Annotated image is not uint8"
    assert not np.all(annot == 0), "Annotated image is completely blank/black"

print("\n=== SUMMARY TABLE ===")
for r in results_table:
    print(f"Image: {r['image']} | Detections: {r['count']} | Classes: {r['classes']} | Confs: {r['confs']} | Time: {r['time_sec']}s")

print("\n=== ALL PHASE 3 INFERENCE CHECKS PASSED ===")
