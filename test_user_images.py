from detector import RoadDamageDetector
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

detector = RoadDamageDetector()
print("Detector status:", detector.get_status())

images = [
    ("Screenshot 1: Potholes_orig (3 Water Puddles/Potholes)", "data/uploads/20261015_095002_907236_Potholes_orig.jpg"),
    ("Screenshot 2: pt-2_orig (Car + Large Foreground Pothole)", "data/uploads/20261015_095354_523402_pt-2_orig.jpg")
]

for title, path in images:
    res = detector.predict(path, conf_threshold=0.25)
    print(f"\n=== {title} ===")
    print(f"Total defects detected : {res['total_detections']}")
    print(f"Overall severity       : {res['overall_severity']}")
    print(f"Inference latency      : {res['inference_time_sec']:.2f}s")
    for d in res['detections']:
        print(f"  -> {d['class_name']}: {d['confidence']*100:.1f}% | Severity: {d['severity']} | Box: {d['box']}")
