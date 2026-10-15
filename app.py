"""
RoadSentinel - AI-Powered Road Damage Detection MVP
College Machine Learning & Tools (MLT) Project Demonstration.
"""

import os
import io
import time
import cv2
import numpy as np
from PIL import Image
import streamlit as st

from detector import RoadDamageDetector, CLASS_NAMES, CLASS_COLORS
from evaluate import load_evaluation_metrics
import history_manager

# Set Streamlit page configuration
st.set_page_config(
    page_title="RoadSentinel — AI Road Damage Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling: Restrained Dark Navy with Emerald / Amber Accents
st.markdown("""
<style>
    /* Dark Navy Theme Styling */
    .stApp {
        background-color: #0b132b;
        color: #e0e6ed;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Header Banner */
    .main-header {
        background: linear-gradient(135deg, #1c2541 0%, #0b132b 100%);
        border: 1px solid #3a506b;
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-tagline {
        color: #9bb1c8;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #1c2541;
        border: 1px solid #3a506b;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #48e5c2;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9bb1c8;
        margin-top: 4px;
    }
    
    /* Severity Badges */
    .severity-badge-high {
        background-color: rgba(230, 57, 70, 0.2);
        color: #ff6b6b;
        border: 1px solid #e63946;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .severity-badge-med {
        background-color: rgba(255, 183, 3, 0.2);
        color: #ffb703;
        border: 1px solid #ffb703;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .severity-badge-low {
        background-color: rgba(72, 229, 194, 0.2);
        color: #48e5c2;
        border: 1px solid #48e5c2;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111a33;
        border-right: 1px solid #283754;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #1c2541;
        border-radius: 8px 8px 0 0;
        color: #9bb1c8;
        border: 1px solid transparent;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #243356 !important;
        color: #ffffff !important;
        border-color: #3a506b #3a506b transparent #3a506b !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Detector (cached in st.session_state)
@st.cache_resource
def get_detector():
    return RoadDamageDetector()

detector = get_detector()
detector_status = detector.get_status()

# ----------------- SIDEBAR: CONTROLS & STATUS -----------------
with st.sidebar:
    st.markdown("### 🛡️ RoadSentinel Control")
    st.markdown("Configure inference parameters and monitor model runtime status.")
    
    st.markdown("---")
    st.markdown("#### ⚙️ Inference Parameters")
    conf_thresh = st.slider(
        "Confidence Threshold", 
        min_value=0.10, 
        max_value=0.90, 
        value=0.25, 
        step=0.05,
        help="Filter predictions with confidence score above this limit."
    )
    
    nms_thresh = st.slider(
        "NMS IoU Threshold", 
        min_value=0.20, 
        max_value=0.80, 
        value=0.45, 
        step=0.05,
        help="Non-Maximum Suppression threshold to merge overlapping boxes."
    )
    
    st.markdown("---")
    st.markdown("#### 🔍 Model Runtime Status")
    if detector_status["loaded"]:
        st.success(f"**Loaded:** {detector_status['architecture']}")
        st.caption(f"**Backend:** {detector_status['backend']}")
        st.caption(f"**Origin:** {detector_status['source']}")
    else:
        st.error(f"**Model Unloaded:** {detector_status['error']}")
        st.warning("Running in degraded mode. Please verify `models/road_damage_yolov8.onnx`.")

    st.markdown("---")
    st.markdown("#### 🏷️ Detected Categories")
    for cls_name in CLASS_NAMES:
        bgr = CLASS_COLORS.get(cls_name, (255, 255, 255))
        rgb = f"rgb({bgr[2]},{bgr[1]},{bgr[0]})"
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;font-size:0.85rem;margin-bottom:4px;'>"
            f"<div style='width:12px;height:12px;border-radius:3px;background-color:{rgb};'></div>"
            f"<span>{cls_name}</span>"
            f"</div>", 
            unsafe_allow_html=True
        )

# ----------------- HEADER -----------------
st.markdown("""
<div class="main-header">
    <div class="header-title">
        <span>🛡️ RoadSentinel</span>
        <span style="font-size: 1rem; background-color: #2a3d66; color: #48e5c2; padding: 4px 10px; border-radius: 20px; font-weight: 500;">MVP Release</span>
    </div>
    <p class="header-tagline">
        Autonomous Deep Learning Inspection Engine for Asphalt Damage, Potholes, and Structural Road Distress.
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------- MAIN TABS -----------------
tab_inspect, tab_history, tab_metrics, tab_about = st.tabs([
    "📸 Damage Inspection", 
    "🕒 Inspection History", 
    "📊 Model Benchmark & Metrics", 
    "ℹ️ Methodology & Architecture"
])

# ----------------- TAB 1: DAMAGE INSPECTION -----------------
with tab_inspect:
    st.markdown("#### 1. Select Road Image Source")
    input_mode = st.radio(
        "Choose how to provide the inspection image:",
        options=[
            "📁 Upload Your Own Road Image", 
            "🛣️ Benchmark Test Dataset (RDD2022 & Kaggle Verified Samples)"
        ],
        horizontal=True
    )

    active_image_bytes = None
    active_filename = None
    source_type_label = None

    if input_mode == "📁 Upload Your Own Road Image":
        uploaded_file = st.file_uploader(
            "Upload Road Image for Inspection", 
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Upload an image from a dashcam, street-level smartphone, or drone road survey."
        )
        if uploaded_file is not None:
            try:
                active_image_bytes = uploaded_file.read()
                active_filename = uploaded_file.name
                source_type_label = f"User Upload: {active_filename}"
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")
        else:
            st.info("👆 Please select or drag & drop a real-world asphalt pavement photograph from your computer.")

    else:
        st.markdown("**Select from Verified Real-World Benchmark Road Damage Images:**")
        real_benchmark_samples = {
            "Select a benchmark sample...": None,
            "Pothole: India_000093 (RDD2022 Verified Pothole - 53.9% conf)": "samples/real_road_damage/pothole/India_000093.jpg",
            "Pothole: India_000092 (RDD2022 Multi-Defect Pothole & Crack)": "samples/real_road_damage/pothole/India_000092.jpg",
            "Pothole: Kaggle Benchmark 01 (Street Asphalt Pothole)": "samples/real_road_damage/pothole/kaggle_pothole_01.jpg",
            "Pothole: Kaggle Benchmark 02 (Street Asphalt Pothole)": "samples/real_road_damage/pothole/kaggle_pothole_02.jpg",
            "Alligator Crack: India_000024 (RDD2022 Base Fatigue - 55.9% conf)": "samples/real_road_damage/alligator_crack/India_000024.jpg",
            "Longitudinal Crack: India_000104 (RDD2022 Linear Crack - 44.6% conf)": "samples/real_road_damage/longitudinal_crack/India_000104.jpg",
            "Longitudinal Crack: India_000125 (RDD2022 Multiple Linear Cracks)": "samples/real_road_damage/longitudinal_crack/India_000125.jpg",
            "Transverse Crack: India_001201 (RDD2022 Thermal Crack - 50.9% conf)": "samples/real_road_damage/transverse_crack/India_001201.jpg",
        }
        selected_bench_key = st.selectbox(
            "Benchmark Samples (CRDDC 2022 / Kaggle)",
            options=list(real_benchmark_samples.keys()),
            index=0
        )
        sample_path = real_benchmark_samples[selected_bench_key]
        if sample_path and os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                active_image_bytes = f.read()
            active_filename = os.path.basename(sample_path)
            source_type_label = f"Verified Benchmark: {selected_bench_key.split('(')[0].strip()}"
        else:
            st.info("👆 Select a verified road damage photograph from the dropdown above to test.")

    if active_image_bytes:
        st.markdown("---")
        
        # Operational context badges
        col_ctx1, col_ctx2, col_ctx3 = st.columns(3)
        with col_ctx1:
            st.caption(f"**Source:** {source_type_label}")
        with col_ctx2:
            st.caption(f"**Loaded Model:** `models/road_damage_yolov8.onnx`")
        with col_ctx3:
            st.caption(f"**Confidence Threshold:** `{conf_thresh:.2f}` (NMS: `{nms_thresh:.2f}`)")

        # Deduplication key to prevent duplicate DB records on UI tab switches
        import hashlib
        img_hash = hashlib.md5(active_image_bytes).hexdigest()
        run_key = f"{img_hash}_{conf_thresh}_{nms_thresh}"
        
        with st.spinner("Executing genuine YOLOv8 neural network inference via OpenCV DNN..."):
            try:
                # Perform genuine detection
                result = detector.predict(
                    active_image_bytes, 
                    conf_threshold=conf_thresh, 
                    nms_threshold=nms_thresh
                )
                
                # Only persist new inspection when input image or thresholds change
                if st.session_state.get("last_saved_run_key") != run_key:
                    record_id = history_manager.save_inspection(
                        active_filename,
                        active_image_bytes,
                        result["annotated_image"],
                        result,
                        conf_thresh
                    )
                    st.session_state["last_saved_run_key"] = run_key
                    st.session_state["last_record_id"] = record_id
                else:
                    record_id = st.session_state.get("last_record_id", history_manager.get_total_inspection_count())
            except Exception as e:
                st.error(f"Inference error: {e}")
                st.stop()

        # Display Top Summary Metrics
        st.markdown("### 📊 Detection Summary")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result['total_detections']}</div>
                <div class="metric-label">Defects Detected</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.4rem;">{result['overall_severity'].split()[0]}</div>
                <div class="metric-label">Severity Level</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result['inference_time_sec']:.2f}s</div>
                <div class="metric-label">Inference Latency</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">#{record_id}</div>
                <div class="metric-label">Saved Inspection ID</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Side-by-Side Visual Comparison
        col_orig, col_annot = st.columns(2)
        
        with col_orig:
            st.markdown("#### 📷 Original Input Image (Unmodified)")
            orig_rgb = cv2.cvtColor(result["original_image"], cv2.COLOR_BGR2RGB)
            st.image(orig_rgb, use_container_width=True, caption=f"Original Input ({active_filename})")
            
        with col_annot:
            st.markdown("#### 🎯 Genuine Model Output (Annotated Detections)")
            annot_rgb = cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
            st.image(annot_rgb, use_container_width=True, caption=f"YOLOv8 Prediction (Conf >= {conf_thresh:.2f})")

        # Detailed Breakdown Table or Zero Detections Notice
        st.markdown("---")
        st.markdown("#### 📋 Detected Road Distress Items")
        if result["total_detections"] == 0:
            st.warning(f"⚠️ No objects detected above the current confidence threshold ({conf_thresh:.2f}). This does not prove that the road is damage-free.")
            st.markdown(f"""
            <div style='background-color:#1c2541; padding:16px 20px; border-radius:8px; border-left:4px solid #ffb703; font-size:0.92rem; margin-top:8px;'>
                <b>🔍 Model Diagnostic Insights:</b>
                <ul style='margin-bottom:6px; padding-left:20px; margin-top:6px;'>
                    <li><b>Asphalt Context Requirement:</b> The YOLOv8 model was trained on vehicle dashcam and street-level asphalt imagery (RDD2022). Isolated 3D graphic cutouts, renderings, or images against pure black voids do not activate real pavement convolutional feature maps.</li>
                    <li><b>Confidence Threshold:</b> Faint, narrow, or partially occluded cracks may activate candidate anchors below the current threshold of <code>{conf_thresh:.2f}</code>. Try lowering the <b>Confidence Threshold slider</b> in the left sidebar to <code>0.15</code> or <code>0.20</code>.</li>
                    <li><b>Pavement Orientation:</b> Ensure the road surface fills the central field of view with adequate lighting.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            table_data = []
            for idx, det in enumerate(result["detections"], 1):
                x, y, w, h = det["box"]
                table_data.append({
                    "Item #": idx,
                    "Damage Class": det["class_name"],
                    "Confidence": f"{det['confidence'] * 100:.1f}%",
                    "Severity": det["severity"],
                    "Bounding Box (X, Y, W, H)": f"({x}, {y}, {w}, {h})"
                })
            st.dataframe(table_data, use_container_width=True)

# ----------------- TAB 2: INSPECTION HISTORY -----------------
with tab_history:
    st.markdown("### 🕒 Recent Inspection Records")
    st.caption("Inspections and annotated visual outputs saved locally in `data/`.")
    
    recent_records = history_manager.get_recent_inspections(limit=15)
    
    if not recent_records:
        st.write("No inspections recorded yet. Run a prediction in the Damage Inspection tab to see records here.")
    else:
        for r in recent_records:
            with st.expander(f"Inspection #{r['id']} — {r['filename']} ({r['timestamp']}) - {r['total_detections']} Defects [{r['severity']}]"):
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    if os.path.exists(r["original_path"]):
                        st.image(r["original_path"], caption="Original Image", use_container_width=True)
                with c2:
                    if os.path.exists(r["annotated_path"]):
                        st.image(r["annotated_path"], caption="Annotated Output", use_container_width=True)
                with c3:
                    st.markdown(f"**Timestamp:** {r['timestamp']}")
                    st.markdown(f"**Total Detections:** {r['total_detections']}")
                    st.markdown(f"**Severity:** {r['severity']}")
                    st.markdown(f"**Inference Time:** {r['inference_time']:.2f}s")
                    st.markdown(f"**Confidence Threshold Used:** {r['conf_threshold']}")
                    if r["class_counts"]:
                        st.markdown("**Damage Breakdown:**")
                        for k, v in r["class_counts"].items():
                            st.write(f"- {k}: {v}")

# ----------------- TAB 3: MODEL BENCHMARK & METRICS -----------------
with tab_metrics:
    st.markdown("### 📊 Official Model Evaluation & Benchmark Metrics")
    st.caption("Computed from the authentic Road Damage Dataset (RDD) validation run across 76 training epochs.")

    eval_data = load_evaluation_metrics()
    
    if eval_data["status"] == "Not evaluated yet":
        st.warning("Model evaluation data is not available.")
    else:
        m = eval_data["metrics"]
        
        # Metric KPI cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{m['mAP50'] * 100:.2f}%</div>
                <div class="metric-label">mAP@0.5 (Overall)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{m['mAP50_95'] * 100:.2f}%</div>
                <div class="metric-label">mAP@0.5:0.95</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{m['precision'] * 100:.2f}%</div>
                <div class="metric-label">Precision (B)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{m['recall'] * 100:.2f}%</div>
                <div class="metric-label">Recall (B)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Technical Validation Specifications
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("#### 🔬 Benchmark Specifications")
            st.markdown(f"- **Benchmark Source:** {eval_data['source']}")
            st.markdown(f"- **Target Dataset:** {eval_data['dataset']}")
            st.markdown(f"- **Architecture Checkpoint:** {eval_data['architecture']}")
            st.markdown(f"- **Total Epochs Trained:** {eval_data['total_epochs_trained']}")
            st.markdown(f"- **Optimal Validation Epoch:** Epoch #{eval_data['best_epoch']}")
            st.markdown(f"- **Calculated F1-Score:** {m['f1_score'] * 100:.2f}%")

        with col_info2:
            st.markdown("#### 📉 Final Validation Loss Breakdown")
            fem = m["final_epoch_metrics"]
            st.markdown(f"- **Validation Bounding Box Loss:** `{fem['val_box_loss']}`")
            st.markdown(f"- **Validation Classification Loss:** `{fem['val_cls_loss']}`")
            st.markdown(f"- **Validation Distribution Focal Loss (DFL):** `{fem['val_dfl_loss']}`")
            st.markdown(f"- **Classes Evaluated:** {len(eval_data['classes_evaluated'])}")

        st.markdown("---")
        st.markdown("#### 🏷️ Evaluated Road Damage Taxonomy")
        st.write(", ".join([f"`{c}`" for c in eval_data["classes_evaluated"]]))

# ----------------- TAB 4: METHODOLOGY & ARCHITECTURE -----------------
with tab_about:
    st.markdown("### ℹ️ RoadSentinel Architecture & Academic Methodology")
    
    st.markdown("""
    #### 1. Problem Statement
    Road degradation (longitudinal cracks, alligator cracks, potholes) presents severe public safety risks, 
    vehicle damage, and economic municipal costs. Manual visual road audits are slow, hazardous, and infrequent.
    **RoadSentinel** provides automated, objective computer-vision damage classification directly from road imagery.

    #### 2. Model & Pipeline Architecture
    - **Backbone & Detection Head:** YOLOv8 (You Only Look Once v8) Object Detection network featuring CSPDarknet backbone and Path Aggregation Network (PAN) neck.
    - **Inference Runtime Engine:** OpenCV DNN C++ engine executing ONNX format graph. This allows reliable, optimized CPU inference with SSE4/AVX instructions without requiring CUDA or heavy framework overhead.
    - **Taxonomy:** 5 specialized classes compliant with the IEEE Global Road Damage Detection Challenge (RDD):
      1. *Longitudinal Crack* (Parallel surface distress)
      2. *Transverse Crack* (Perpendicular thermal/joint distress)
      3. *Alligator Crack* (Fatigue structural degradation)
      4. *Pothole* (Severe cavity distress requiring urgent repair)
      5. *Other Road Damage* (General pavement anomalies)
    - **Severity Assessment Logic:**
      - Potholes & Alligator Cracks are categorized as **High Severity** due to immediate accident potential and structural sub-base damage.
      - Longitudinal & Transverse Cracks are categorized as **Medium Severity** (routine sealing maintenance).

    #### 3. Hardware & Environmental Disclosures (College Submission Integrity)
    - **Current Host:** Intel(R) Core(TM) i5-3210M CPU (Ivy Bridge architecture).
    - **Execution Strategy:** Because Ivy Bridge CPUs support AVX 1.0 but not AVX2, the runtime engine deploys the model via OpenCV's C++ DNN ONNX runtime. This guarantees 100% genuine execution with zero simulated outputs.
    - **Fine-Tuning:** The fine-tuning pipeline is reproducible via `python train.py`. Full heavy re-training requires a CUDA GPU or cloud compute (Google Colab / Kaggle).
    """)
