"""
RoadSentinel - End-to-End Workflow & Integration Test Suite
Validates model loading, inference accuracy, image processing, database persistence, and evaluation metrics.
"""

import os
import sys
import cv2
import json
import urllib.request

def run_tests():
    print("=" * 70)
    print("        ROADSENTINEL - AUTOMATED END-TO-END VERIFICATION       ")
    print("=" * 70)
    
    passed_tests = 0
    total_tests = 6

    # Test 1: Check Model Weights Exist & Size
    print("\n[TEST 1/6] Verifying Model Artifacts...")
    model_path = "models/road_damage_yolov8.onnx"
    if os.path.exists(model_path) and os.path.getsize(model_path) > 10_000_000:
        print(f" PASS: Model weights verified ({os.path.getsize(model_path):,} bytes).")
        passed_tests += 1
    else:
        print(" FAIL: Model weights missing or corrupt.")

    # Test 2: Detector Loading and Backend Verification
    print("\n[TEST 2/6] Verifying Detector Initialization...")
    from detector import RoadDamageDetector, CLASS_NAMES
    detector = RoadDamageDetector()
    status = detector.get_status()
    if status["loaded"] and len(status["classes"]) == 5:
        print(f" PASS: Detector initialized with backend '{status['backend']}'.")
        print(f"       Recognized classes: {status['classes']}")
        passed_tests += 1
    else:
        print(f" FAIL: Detector load error: {status.get('error')}")

    # Test 3: Genuine Inference on Sample Images
    print("\n[TEST 3/6] Running Genuine Inference on Sample Images...")
    test_image = "samples/featured/alligator_crack_sample.jpg"
    if os.path.exists(test_image):
        pred = detector.predict(test_image, conf_threshold=0.20)
        if pred["success"] and pred["total_detections"] > 0:
            print(f" PASS: Inference succeeded in {pred['inference_time_sec']:.3f}s.")
            print(f"       Detections found: {pred['total_detections']} (Severity: {pred['overall_severity']})")
            for d in pred["detections"]:
                print(f"        -> {d['class_name']}: {d['confidence']*100:.1f}% conf, Box: {d['box']}")
            passed_tests += 1
        else:
            print(f" FAIL: Inference returned 0 detections on validated sample.")
    else:
        print(f" FAIL: Sample image not found at {test_image}")

    # Test 4: Image Annotation & Rendering Verification
    print("\n[TEST 4/6] Verifying Annotated Image Generation...")
    if "annotated_image" in pred and pred["annotated_image"] is not None:
        h, w = pred["annotated_image"].shape[:2]
        if h > 0 and w > 0:
            print(f" PASS: Annotated image generated with shape ({h}, {w}, 3).")
            passed_tests += 1
        else:
            print(" FAIL: Annotated image has invalid shape.")
    else:
        print(" FAIL: Annotated image missing from prediction output.")

    # Test 5: Inspection History Persistence
    print("\n[TEST 5/6] Verifying Local SQLite Inspection History...")
    import history_manager
    count_before = history_manager.get_total_inspection_count()
    with open(test_image, "rb") as f:
        img_bytes = f.read()
    rec_id = history_manager.save_inspection(
        "test_run.jpg", img_bytes, pred["annotated_image"], pred, 0.20
    )
    count_after = history_manager.get_total_inspection_count()
    recent = history_manager.get_recent_inspections(1)
    if count_after > count_before and recent and recent[0]["id"] == rec_id:
        print(f" PASS: Inspection #{rec_id} saved and retrieved from SQLite successfully.")
        print(f"       Timestamp: {recent[0]['timestamp']}, Filename: {recent[0]['filename']}")
        passed_tests += 1
    else:
        print(" FAIL: Failed to persist or retrieve inspection from history.")

    # Test 6: Model Evaluation Metrics Module
    print("\n[TEST 6/6] Verifying Model Evaluation Benchmark...")
    from evaluate import load_evaluation_metrics
    eval_metrics = load_evaluation_metrics()
    if eval_metrics["status"] != "Not evaluated yet" and "metrics" in eval_metrics:
        m = eval_metrics["metrics"]
        print(f" PASS: Benchmark metrics parsed successfully:")
        print(f"       mAP@0.5: {m['mAP50']*100:.2f}% | Precision: {m['precision']*100:.2f}% | Recall: {m['recall']*100:.2f}%")
        passed_tests += 1
    else:
        print(" FAIL: Evaluation metrics not available.")

    print("\n" + "=" * 70)
    print(f"RESULT: {passed_tests}/{total_tests} Tests Passed.")
    print("=" * 70)
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
