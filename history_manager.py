"""
RoadSentinel - Local Inspection History Store
Persists inspection records and image artifacts with SQLite or JSON fallback.
"""

import os
import json
import sqlite3
import datetime
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join("data", "inspections.db")
UPLOAD_DIR = os.path.join("data", "uploads")
OUTPUT_DIR = os.path.join("data", "outputs")

os.makedirs("data", exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            original_path TEXT NOT NULL,
            annotated_path TEXT NOT NULL,
            total_detections INTEGER NOT NULL,
            severity TEXT NOT NULL,
            inference_time REAL NOT NULL,
            conf_threshold REAL NOT NULL,
            detections_json TEXT NOT NULL,
            class_counts_json TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_inspection(
    filename: str,
    orig_img_bytes: bytes,
    annotated_img_bgr,
    results: Dict[str, Any],
    conf_threshold: float
) -> int:
    """Save inspection record, original and annotated images to disk and DB."""
    import cv2
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    orig_filename = f"{timestamp_str}_{base_name}_orig.jpg"
    annot_filename = f"{timestamp_str}_{base_name}_pred.jpg"
    
    orig_path = os.path.join(UPLOAD_DIR, orig_filename)
    annot_path = os.path.join(OUTPUT_DIR, annot_filename)
    
    # Save original
    with open(orig_path, "wb") as f:
        f.write(orig_img_bytes)
        
    # Save annotated
    cv2.imwrite(annot_path, annotated_img_bgr)
    
    # Save record to SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO inspections (
            timestamp, filename, original_path, annotated_path, 
            total_detections, severity, inference_time, conf_threshold, 
            detections_json, class_counts_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        filename,
        orig_path,
        annot_path,
        results.get("total_detections", 0),
        results.get("overall_severity", "Unknown"),
        results.get("inference_time_sec", 0.0),
        conf_threshold,
        json.dumps(results.get("detections", [])),
        json.dumps(results.get("class_counts", {}))
    ))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_recent_inspections(limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve the most recent inspections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM inspections 
        ORDER BY id DESC 
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "filename": r["filename"],
            "original_path": r["original_path"],
            "annotated_path": r["annotated_path"],
            "total_detections": r["total_detections"],
            "severity": r["severity"],
            "inference_time": r["inference_time"],
            "conf_threshold": r["conf_threshold"],
            "detections": json.loads(r["detections_json"]),
            "class_counts": json.loads(r["class_counts_json"])
        })
    conn.close()
    return results

def get_total_inspection_count() -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM inspections")
    count = cursor.fetchone()[0]
    conn.close()
    return count
