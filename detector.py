"""
RoadSentinel - Computer Vision Inference Engine
Autonomous Road Damage & Pothole Detection using YOLOv8 ONNX and OpenCV DNN.
Combines dedicated Pothole Detection with Multi-Class Road Surface Distress Detection.
"""

import os
import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional

# Authentic class mapping from Road Damage Datasets
CLASS_NAMES = [
    "Longitudinal Crack",  # 0
    "Transverse Crack",    # 1
    "Alligator Crack",     # 2
    "Pothole",             # 3
    "Other Road Damage"    # 4
]

# Color palette (BGR for OpenCV): Distinct colors for each damage class
CLASS_COLORS = {
    "Longitudinal Crack": (255, 165, 0),    # Amber
    "Transverse Crack": (255, 105, 180),    # Violet / Pink
    "Alligator Crack": (0, 140, 255),      # Orange
    "Pothole": (0, 0, 230),                # Vivid Red
    "Other Road Damage": (50, 205, 50)     # Lime Green
}

# Severity classification based on damage type and confidence
SEVERITY_LEVELS = {
    "Pothole": "High",
    "Alligator Crack": "High",
    "Longitudinal Crack": "Medium",
    "Transverse Crack": "Medium",
    "Other Road Damage": "Low"
}

class RoadDamageDetector:
    def __init__(
        self, 
        model_path: str = "models/road_damage_yolov8.onnx",
        pothole_model_path: str = "models/pothole_yolov8.onnx"
    ):
        self.model_path = model_path
        self.pothole_model_path = pothole_model_path
        self.net = None
        self.net_pothole = None
        self.is_loaded = False
        self.load_error = None
        self.input_size = (640, 640)
        self.model_source = "YOLOv8 Unified Ensemble (Dedicated Pothole Head + RDD2022 Surface Distress)"
        self.load_model()

    def load_model(self) -> bool:
        """Load ONNX weights into OpenCV DNN engine."""
        errors = []
        
        # 1. Primary RDD Model
        if os.path.exists(self.model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
            except Exception as e:
                errors.append(f"RDD model load error: {e}")
        else:
            errors.append(f"Model file not found: {self.model_path}")

        # 2. Dedicated Pothole Model
        if os.path.exists(self.pothole_model_path):
            try:
                self.net_pothole = cv2.dnn.readNetFromONNX(self.pothole_model_path)
            except Exception as e:
                errors.append(f"Pothole model load error: {e}")
        else:
            errors.append(f"Pothole model not found: {self.pothole_model_path}")

        if self.net is not None or self.net_pothole is not None:
            self.is_loaded = True
            self.load_error = "; ".join(errors) if errors else None
            return True
        else:
            self.is_loaded = False
            self.load_error = "; ".join(errors)
            return False

    def get_status(self) -> Dict[str, Any]:
        """Return operational status of the model."""
        return {
            "loaded": self.is_loaded,
            "model_path": self.model_path,
            "pothole_model_path": self.pothole_model_path,
            "architecture": "YOLOv8 Small (ONNX Dual Engine)",
            "backend": "OpenCV DNN C++ Engine",
            "source": self.model_source,
            "classes": CLASS_NAMES,
            "error": self.load_error
        }

    def predict(
        self, 
        image_input, 
        conf_threshold: float = 0.25, 
        nms_threshold: float = 0.45
    ) -> Dict[str, Any]:
        """
        Run genuine object detection on an image.
        Accepts file path (str), numpy array (BGR/RGB), or bytes.
        """
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError(f"Detector is not loaded: {self.load_error}")

        start_time = time.time()
        
        # Read or convert image
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image path does not exist: {image_input}")
            orig_img = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)
            if orig_img is None:
                raise ValueError("Failed to decode image from path.")
        elif isinstance(image_input, np.ndarray):
            orig_img = image_input.copy()
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            orig_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if orig_img is None:
                raise ValueError("Failed to decode image from raw bytes.")
        else:
            raise TypeError("Unsupported image input type.")

        # If RGBA (4 channels), blend transparent background into neutral asphalt gray
        if len(orig_img.shape) == 3 and orig_img.shape[2] == 4:
            b, g, r, a = cv2.split(orig_img)
            alpha = (a.astype(float) / 255.0)[..., np.newaxis]
            rgb = cv2.merge([b, g, r]).astype(float)
            bg = np.full_like(rgb, 120.0)
            orig_img = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        elif len(orig_img.shape) == 2:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)

        h_orig, w_orig = orig_img.shape[:2]

        # Prepare YOLOv8 640x640 normalized blob
        blob = cv2.dnn.blobFromImage(
            orig_img, 
            scalefactor=1.0 / 255.0, 
            size=self.input_size, 
            swapRB=True, 
            crop=False
        )

        scale_x = w_orig / float(self.input_size[0])
        scale_y = h_orig / float(self.input_size[1])

        boxes: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        # 1. Run Dedicated Pothole Model if available
        if self.net_pothole is not None:
            self.net_pothole.setInput(blob)
            p_output = self.net_pothole.forward()
            preds_p = np.transpose(p_output[0], (1, 0)) # (8400, 5) -> cx, cy, w, h, conf
            
            for row in preds_p:
                conf = float(row[4])
                if conf >= conf_threshold:
                    cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    left = int((cx - w / 2.0) * scale_x)
                    top = int((cy - h / 2.0) * scale_y)
                    width = int(w * scale_x)
                    height = int(h * scale_y)

                    left = max(0, left)
                    top = max(0, top)
                    width = min(w_orig - left, width)
                    height = min(h_orig - top, height)

                    if width > 6 and height > 6:
                        boxes.append([left, top, width, height])
                        confidences.append(conf)
                        class_ids.append(3) # Pothole ID

        # 2. Run RDD Multi-Class Surface Distress Model if available
        if self.net is not None:
            self.net.setInput(blob)
            rdd_output = self.net.forward()
            preds_rdd = np.transpose(rdd_output[0], (1, 0)) # (8400, 9)

            for row in preds_rdd:
                scores = row[4:]
                cls_id = int(np.argmax(scores))
                conf = float(scores[cls_id])
                
                # If pothole model is already loaded, only take cracks and distress from RDD
                # to prevent lower-confidence RDD mislabeling potholes as 'Other'
                if self.net_pothole is not None and cls_id in (3, 4) and conf < 0.65:
                    continue

                if conf >= conf_threshold:
                    cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    left = int((cx - w / 2.0) * scale_x)
                    top = int((cy - h / 2.0) * scale_y)
                    width = int(w * scale_x)
                    height = int(h * scale_y)

                    left = max(0, left)
                    top = max(0, top)
                    width = min(w_orig - left, width)
                    height = min(h_orig - top, height)

                    if width > 6 and height > 6:
                        boxes.append([left, top, width, height])
                        confidences.append(conf)
                        class_ids.append(cls_id)

        # Apply Non-Maximum Suppression across all detections
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        
        detections: List[Dict[str, Any]] = []
        annotated_img = orig_img.copy()
        class_counts: Dict[str, int] = {c: 0 for c in CLASS_NAMES}

        if len(indices) > 0:
            for idx in indices.flatten():
                box = boxes[idx]
                cls_id = class_ids[idx]
                conf = confidences[idx]
                class_name = CLASS_NAMES[cls_id]
                severity = SEVERITY_LEVELS.get(class_name, "Low")
                
                class_counts[class_name] += 1
                detections.append({
                    "box": box, # [x, y, w, h]
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": float(conf),
                    "severity": severity
                })

                # Draw high-contrast bounding box
                color = CLASS_COLORS.get(class_name, (0, 255, 0))
                x, y, w, h = box
                
                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 3)

                label = f"{class_name} {conf:.1%}"
                (lbl_w, lbl_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                tag_y = max(y, lbl_h + 10)
                cv2.rectangle(
                    annotated_img, 
                    (x, tag_y - lbl_h - 8), 
                    (x + lbl_w + 10, tag_y + baseline - 2), 
                    color, 
                    -1
                )
                cv2.putText(
                    annotated_img, 
                    label, 
                    (x + 5, tag_y - 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    2, 
                    cv2.LINE_AA
                )

        latency = time.time() - start_time

        # Calculate overall road severity
        overall_severity = "Clear / Good"
        if any(d["severity"] == "High" for d in detections):
            overall_severity = "High Severity (Urgent Repair Required)"
        elif any(d["severity"] == "Medium" for d in detections):
            overall_severity = "Medium Severity (Maintenance Scheduled)"
        elif len(detections) > 0:
            overall_severity = "Low Severity (Surface Distress)"

        return {
            "success": True,
            "inference_time_sec": latency,
            "total_detections": len(detections),
            "detections": detections,
            "class_counts": {k: v for k, v in class_counts.items() if v > 0},
            "overall_severity": overall_severity,
            "annotated_image": annotated_img,
            "original_image": orig_img,
            "image_dimensions": {"width": w_orig, "height": h_orig}
        }
