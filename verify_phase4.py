import onnx
import csv
import json
import os
import sys

# Ensure UTF-8 output on Windows terminal
sys.stdout.reconfigure(encoding="utf-8")

print("=== PHASE 4: MODEL IDENTITY & EVALUATION AUDIT ===")

# 1. Inspect ONNX model graph & metadata
model_path = "models/road_damage_yolov8.onnx"
onnx_model = onnx.load(model_path)
print(f"ONNX IR Version: {onnx_model.ir_version}")
print(f"ONNX Producer: {onnx_model.producer_name} {onnx_model.producer_version}")
print("ONNX Metadata Properties:")
for prop in onnx_model.metadata_props:
    print(f"  {prop.key}: {prop.value}")

graph = onnx_model.graph
for inp in graph.input:
    shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
    print(f"  Input '{inp.name}' shape: {shape}")

for out in graph.output:
    shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
    print(f"  Output '{out.name}' shape: {shape}")

# 2. Inspect assets/results.csv
csv_path = "assets/results.csv"
assert os.path.exists(csv_path), "results.csv missing"
print(f"\n--- Inspecting {csv_path} ---")
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

print(f"Total Rows (Epochs): {len(rows)}")
print("Columns in results.csv:")
for idx, col in enumerate(header):
    print(f"  [{idx:2d}] {col}")

# Check first and last row
first_row = {header[i].strip(): rows[0][i].strip() for i in range(len(header))}
last_row = {header[i].strip(): rows[-1][i].strip() for i in range(len(header))}
print(f"\nEpoch 1: Epoch={first_row.get('epoch')}, P={first_row.get('metrics/precision(B)')}, R={first_row.get('metrics/recall(B)')}, mAP50={first_row.get('metrics/mAP50(B)')}, mAP50-95={first_row.get('metrics/mAP50-95(B)')}")
print(f"Epoch {len(rows)}: Epoch={last_row.get('epoch')}, P={last_row.get('metrics/precision(B)')}, R={last_row.get('metrics/recall(B)')}, mAP50={last_row.get('metrics/mAP50(B)')}, mAP50-95={last_row.get('metrics/mAP50-95(B)')}")

# Find best epoch across all rows
map50_col_idx = [i for i, col in enumerate(header) if "metrics/mAP50(B)" in col][0]
map50_vals = [float(r[map50_col_idx]) for r in rows]
best_idx = map50_vals.index(max(map50_vals))
best_row = {header[i].strip(): rows[best_idx][i].strip() for i in range(len(header))}
print(f"\nPeak mAP50 Epoch: Epoch={best_row.get('epoch')}, P={best_row.get('metrics/precision(B)')}, R={best_row.get('metrics/recall(B)')}, mAP50={best_row.get('metrics/mAP50(B)')}, mAP50-95={best_row.get('metrics/mAP50-95(B)')}")

# 3. Check evaluate.py logic
print("\n--- Inspecting evaluate.py Implementation ---")
with open("evaluate.py", "r", encoding="utf-8") as f:
    eval_code = f.read()
    if "load_evaluation_metrics" in eval_code and "results.csv" in eval_code:
        print("evaluate.py explicitly parses and loads historical training run metrics from assets/results.csv.")

print("\n=== PHASE 4 AUDIT COMPLETE ===")
