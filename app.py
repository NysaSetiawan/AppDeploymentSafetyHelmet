import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import requests
import numpy as np

st.set_page_config(page_title="Deteksi Helm Proyek", page_icon="🛡️", layout="wide")

MODEL_CONFIG = {
    "YoloV8": {"filename": "best8.pt", "url": "https://github.com/NysaSetiawan/AppDeploymentSafetyHelmet/releases/download/yolov8/best8.pt"},
    "YoloV11": {"filename": "best11.pt", "url": "https://raw.githubusercontent.com/NysaSetiawan/AppDeploymentSafetyHelmet/main/best11.pt"},
    "YoloV12": {"filename": "best12.pt", "url": "https://raw.githubusercontent.com/NysaSetiawan/AppDeploymentSafetyHelmet/main/best12.pt"},
}

def ensure_models_downloaded():
    for name, config in MODEL_CONFIG.items():
        if not os.path.exists(config["filename"]):
            with st.spinner(f"Mengunduh {name} (hanya dilakukan sekali)..."):
                try:
                    r = requests.get(config["url"], stream=True)
                    r.raise_for_status()
                    with open(config["filename"], "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                except Exception as e:
                    st.error(f"Gagal mengunduh {name}: {e}")

ensure_models_downloaded()

@st.cache_resource
def load_model(path: str):
    return YOLO(path)

def run_detection(model, image: Image.Image):
    results = model.predict(image, classes=[0, 1], conf=0.25, verbose=False)
    plotted = results[0].plot()[:, :, ::-1]  

    boxes = results[0].boxes
    conf_list = boxes.conf.tolist() if boxes is not None else []
    n_det = len(conf_list)
    avg_conf = float(np.mean(conf_list)) * 100 if conf_list else 0.0

    return {
        "image": plotted,
        "n_det": n_det,
        "avg_conf": avg_conf,
        "labels": [results[0].names[int(c)] for c in boxes.cls.tolist()] if boxes is not None else []
    }

def score_model(result: dict) -> float:
    return result["avg_conf"] + (result["n_det"] * 2.0)

st.title("🛡️ Deteksi Hard Hat & Head")
st.caption("Upload gambar untuk mendeteksi pemakaian helm pekerja.")

col_config1, col_config2 = st.columns(2)
with col_config1:
    num_models = st.radio("Jumlah model perbandingan:", [1, 2, 3], horizontal=True)

with col_config2:
    if num_models == 1:
        selected_names = [st.selectbox("Pilih model:", list(MODEL_CONFIG.keys()))]
    elif num_models == 2:
        m1 = st.selectbox("Model 1:", list(MODEL_CONFIG.keys()), index=0)
        remaining = [m for m in MODEL_CONFIG.keys() if m != m1]
        m2 = st.selectbox("Model 2:", remaining)
        selected_names = [m1, m2]
    else:
        selected_names = list(MODEL_CONFIG.keys())
        st.write("Menggunakan semua model (V8, V11, V12)")

uploaded_file = st.file_uploader("Upload gambar (JPG/PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    if st.button("Mulai Deteksi", type="primary"):
        results = {}
        for name in selected_names:
            with st.spinner(f"Menjalankan {name}..."):
                model = load_model(MODEL_CONFIG[name]["filename"])
                results[name] = run_detection(model, image)
        
        st.divider()
        st.subheader("📊 Hasil Deteksi")
        
        cols = st.columns(len(selected_names))
        scores = {name: score_model(res) for name, res in results.items()}
        best_name = max(scores, key=scores.get)
        
        for col, name in zip(cols, selected_names):
            with col:
                is_best = (name == best_name)
                st.markdown(f"**{name}** {'🏆' if is_best else ''}")
                st.image(results[name]["image"], use_column_width=True)
                st.metric("Deteksi", results[name]["n_det"])
                st.metric("Confidence", f"{results[name]['avg_conf']:.1f}%")
