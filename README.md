# 🛡️ RoadSentinel — AI-Powered Road Damage Detection

**College Machine Learning & Tools (MLT) Project**  
*Autonomous Deep Learning Computer Vision System for Automated Pavement Distress Classification*

---

## 📌 Executive Summary

Road infrastructure degradation—such as potholes and structural cracking—presents critical public safety hazards and substantial economic liabilities. Traditional manual road inspection relies on slow, hazardous, and infrequent visual surveys.

**RoadSentinel** is an end-to-end computer-vision system that automates road distress identification. Using a specialized **YOLOv8** model trained on the international **Road Damage Dataset (RDD)**, the system performs real-time detection and severity categorization of asphalt anomalies directly from digital imagery.

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
Clone this repository and ensure Python is installed:

```bash
git clone https://github.com/Chirag-Patel123/RoadSentinel.git
cd RoadSentinel
pip install -r requirements.txt
```

### 2. Launching the Web Interface
Run the Streamlit application:

```bash
streamlit run app.py
```
Once started, the application opens in your default browser at `http://localhost:8501`.

### 3. Running Model Evaluation
To verify and print the authentic model evaluation metrics:

```bash
python evaluate.py
```

### 4. Running Real Image Benchmark Test Suite
To audit model inference across 11 genuine RDD2022 and Kaggle road damage test images:

```bash
python test_real_images.py 0.20
```

### 5. Running Model Fine-Tuning Pipeline
To initiate the reproducible fine-tuning pipeline on custom datasets:

```bash
python train.py --data data.yaml --epochs 50 --imgsz 640
```

---

## 🎯 Features Built for MVP

1. **Interactive Web Dashboard:** Clean, responsive dark navy UI with real-time detection controls.
2. **Flexible Image Input:** Upload any standard image format (`.jpg`, `.jpeg`, `.png`, `.webp`) or select from bundled demo samples.
3. **Genuine YOLOv8 Object Detection:** Runs inference on road pavement images using OpenCV DNN C++ ONNX engine.
4. **Visual Damage Localization:** Renders color-coded bounding boxes and label tags directly on the pavement surface.
5. **Adjustable Inference Thresholds:** Interactive sliders for Confidence Threshold (0.10–0.90) and Non-Maximum Suppression (NMS) IoU (0.20–0.80).
6. **Damage Severity Classification:** Real-time risk grading (High / Medium / Low) based on damage type and confidence.
7. **Local Inspection History:** Automatically persists inspections, timestamped originals, annotated outputs, and metadata into a local SQLite database (`data/inspections.db`).
8. **Benchmark & Evaluation Tab:** Direct transparency displaying official RDD validation metrics.

---

## 🏷️ Detected Categories & Severity Mapping

| Class ID | Damage Category | Description | Severity Level |
|---|---|---|---|
| **0** | **Longitudinal Crack** | Parallel fatigue cracks along direction of traffic | **Medium** |
| **1** | **Transverse Crack** | Perpendicular cracks from thermal shrinkage / joints | **Medium** |
| **2** | **Alligator Crack** | Interconnected polygon cracking from base failure | **High** |
| **3** | **Pothole** | Bowl-shaped structural depression in road surface | **High** |
| **4** | **Other Road Damage** | General surface raveling, patching, or minor anomalies | **Low** |

---

## 📊 Authentic Model Evaluation Metrics

Evaluated on the official validation split of the Road Damage Dataset (RDD):

- **Mean Average Precision (mAP@0.5):** `62.85%`
- **Mean Average Precision (mAP@0.5:0.95):** `34.46%`
- **Precision (B):** `66.20%`
- **Recall (B):** `58.28%`
- **Calculated F1-Score:** `61.99%`
- **Training Epochs:** 76 Epochs
- **Validation Box Loss:** `1.6403`
- **Validation Class Loss:** `1.4754`
- **Validation DFL Loss:** `1.4438`

---

## 📁 Repository Structure

```
RoadSentinel/
├── app.py                     # Streamlit frontend & interactive dashboard
├── detector.py                # Core YOLOv8 inference engine (OpenCV DNN ONNX)
├── evaluate.py                # Model evaluation and metric reporting script
├── train.py                   # Reproducible YOLO fine-tuning pipeline
├── history_manager.py         # SQLite inspection persistence & history store
├── requirements.txt           # Python package dependencies
├── models/
│   └── road_damage_yolov8.onnx # Trained Road Damage YOLOv8 weights (ONNX format)
├── assets/
│   └── results.csv            # Official training & evaluation metric logs
├── data/
│   ├── inspections.db         # Local SQLite inspection database
│   ├── uploads/               # Saved original uploaded images
│   └── outputs/               # Saved annotated prediction outputs
├── outputs/
│   └── evaluation_report.json # Serialized evaluation benchmark report
└── samples/
    └── featured/              # Curated road damage sample images for quick testing
```

---

## ⚖️ Academic Submission Disclosures

1. **Model Provenance:** The deployed model is a pretrained YOLOv8-small checkpoint fine-tuned on the IEEE Global Road Damage Detection Dataset (RDD2022). It is not a generic COCO model relabeled for road damage.
2. **Runtime Environment:** Inference runs locally on CPU using OpenCV's optimized C++ DNN ONNX runtime. This ensures robust cross-platform execution on standard laptop hardware without requiring dedicated NVIDIA CUDA GPUs.
3. **Training Disclosures:** Full re-training requires an active CUDA GPU environment. The fine-tuning script (`train.py`) is provided for full reproducibility on GPU-equipped workstations or Google Colab.
