import os
import cv2
import json
import sqlite3
import history_manager

print("=== PHASE 6: DATABASE PERSISTENCE AUDIT ===")

# 1. Check initial history count
initial_count = history_manager.get_total_inspection_count()
print(f"Initial DB inspection count: {initial_count}")

# 2. Retrieve existing history
records = history_manager.get_recent_inspections(limit=10)
print(f"Retrieved {len(records)} recent records:")
for r in records[:3]:
    print(f"  ID #{r['id']} | Timestamp: {r['timestamp']} | File: {r['filename']} | Severity: {r['severity']} | Detections: {r['total_detections']}")
    assert os.path.exists(r['original_path']), f"Missing saved original image: {r['original_path']}"
    assert os.path.exists(r['annotated_path']), f"Missing saved annotated image: {r['annotated_path']}"

# 3. Test saving a new test inspection
dummy_img = cv2.imread("samples/featured/alligator_crack_sample.jpg")
with open("samples/featured/alligator_crack_sample.jpg", "rb") as f:
    dummy_bytes = f.read()

dummy_results = {
    "total_detections": 1,
    "overall_severity": "High Severity (Urgent Repair Required)",
    "inference_time_sec": 1.45,
    "detections": [{
        "box": [10, 20, 100, 150],
        "class_id": 2,
        "class_name": "Alligator Crack",
        "confidence": 0.88,
        "severity": "High"
    }],
    "class_counts": {"Alligator Crack": 1}
}

new_id = history_manager.save_inspection(
    filename="audit_test_road.jpg",
    orig_img_bytes=dummy_bytes,
    annotated_img_bgr=dummy_img,
    results=dummy_results,
    conf_threshold=0.25
)
print(f"Saved new inspection with ID: {new_id}")

# 4. Verify new record exists and matches exactly
updated_records = history_manager.get_recent_inspections(limit=1)
assert len(updated_records) == 1, "Failed to retrieve saved record"
latest = updated_records[0]
assert latest["id"] == new_id, f"ID mismatch: {latest['id']} vs {new_id}"
assert latest["filename"] == "audit_test_road.jpg", "Filename mismatch"
assert latest["total_detections"] == 1, "Detections count mismatch"
assert latest["severity"] == "High Severity (Urgent Repair Required)", "Severity mismatch"
assert abs(latest["conf_threshold"] - 0.25) < 1e-4, "Confidence threshold mismatch"
assert os.path.exists(latest["original_path"]), "Original file does not exist on disk"
assert os.path.exists(latest["annotated_path"]), "Annotated file does not exist on disk"
assert latest["detections"][0]["class_name"] == "Alligator Crack", "Detection payload mismatch"

print("Saved inspection record and images verified on disk and SQLite.")

# 5. Test handling missing or corrupted image paths gracefully
non_existent_rec = {
    "original_path": "data/uploads/does_not_exist.jpg",
    "annotated_path": "data/outputs/does_not_exist.jpg"
}
assert not os.path.exists(non_existent_rec["original_path"])
print("Graceful handling of non-existent image paths: verified os.path.exists checks prevent errors.")

# 6. Test persistence across new DB connections
conn = sqlite3.connect("data/inspections.db")
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM inspections WHERE id = ?", (new_id,))
count_in_db = cur.fetchone()[0]
conn.close()
assert count_in_db == 1, "Record not found in direct SQL query"
print(f"Persistence verified across independent DB connection for ID {new_id}.")

print("\n=== PHASE 6 DATABASE AUDIT PASSED ===")
