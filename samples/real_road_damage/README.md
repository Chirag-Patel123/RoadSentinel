# Genuine Road Damage Test Dataset

This directory contains verified, real-world road pavement damage photographs sourced from benchmark road infrastructure datasets with original annotations and licenses.

---

## 1. Primary Dataset: Crowdsensing-based Road Damage Detection (RDD2022)

- **Source:** [IEEE BigData Global Road Damage Detection Challenge (CRDDC 2022)](https://github.com/sekilab/RoadDamageDetector) / [Hugging Face dronefreak/RDD2022](https://huggingface.co/datasets/dronefreak/RDD2022)
- **Paper:** *Crowdsensing-based Road Damage Detection Challenge 2022*, Maeda et al., University of Tokyo / Seki Lab.
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- **Data Capture:** Smartphone-mounted vehicle camera capturing actual asphalt roads in India and Japan.
- **Label Mapping:**
  - Class `0`: `longitudinal_crack` (RDD code D00)
  - Class `1`: `transverse_crack` (RDD code D10)
  - Class `2`: `alligator_crack` (RDD code D20)
  - Class `3`: `pothole` (RDD code D40)

### Verified RDD2022 Images in this Suite:

| Filename | Directory | Verified Ground Truth Label | Original Annotations |
|---|---|---|---|
| `India_000092.jpg` | `samples/real_road_damage/pothole/` | Pothole (Class 3) & Longitudinal Crack (Class 0) | 4 bounding boxes |
| `India_000093.jpg` | `samples/real_road_damage/pothole/` | Pothole (Class 3) | 1 bounding box |
| `India_000104.jpg` | `samples/real_road_damage/longitudinal_crack/` | Longitudinal Crack (Class 0) | 1 bounding box |
| `India_000125.jpg` | `samples/real_road_damage/longitudinal_crack/` | Longitudinal Crack (Class 0) | 4 bounding boxes |
| `India_001201.jpg` | `samples/real_road_damage/transverse_crack/` | Transverse Crack (Class 1) | 1 bounding box |
| `India_001782.jpg` | `samples/real_road_damage/transverse_crack/` | Transverse Crack (Class 1) | 2 bounding boxes |
| `India_000024.jpg` | `samples/real_road_damage/alligator_crack/` | Alligator Crack (Class 2) | 1 bounding box |
| `India_000067.jpg` | `samples/real_road_damage/alligator_crack/` | Alligator Crack (Class 2) | 1 bounding box |

---

## 2. Secondary Dataset: Kaggle / IVCNZ Benchmark Pothole Dataset

- **Source:** [Pothole Detection for YOLO by Anugrah Akbar Praramadhan](https://www.kaggle.com/datasets/anugrahakbar/potholes-detection-for-yolov4) / [GitHub michelpf/dataset-pothole](https://github.com/michelpf/dataset-pothole)
- **License:** Open Data Commons Attribution License (ODC-By) / Public Educational
- **Data Capture:** Street-level asphalt potholes.

### Verified Kaggle Pothole Images:

| Filename | Directory | Verified Ground Truth Label | Source Reference |
|---|---|---|---|
| `kaggle_pothole_01.jpg` | `samples/real_road_damage/pothole/` | Pothole (Class 0 in source) | `01_jpg.rf.3ca97922...jpg` |
| `kaggle_pothole_02.jpg` | `samples/real_road_damage/pothole/` | Pothole (Class 0 in source) | `02_jpg.rf.fd1071cc...jpg` |
| `kaggle_pothole_03.jpg` | `samples/real_road_damage/pothole/` | Pothole (Class 0 in source) | `03_jpg.rf.e6dab082...jpg` |
