import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import requests
import numpy as np

st.set_page_config(
    page_title="Deteksi Helm Proyek",
    page_icon="🛡️",
    layout="wide"
)

MODEL_CONFIG = {
    "YoloV8": {
        "filename": "best8.pt",
        "url": "https://raw.githubusercontent.com/NysaSetiawan/AppDeploymentSafetyHelmet/main/best8.pt",
    },
    "YoloV11": {
        "filename": "best11.pt",
        "url": "https://raw.githubusercontent.com/NysaSetiawan/AppDeploymentSafetyHelmet/main/best11.pt",
    },
    "YoloV12": {
        "filename": "best12.pt",
        "url": "https://raw.githubusercontent.com/NysaSetiawan/AppDeploymentSafetyHelmet/main/best12.pt",
    },
}

def download_model(name: str, config: dict):
    path = config["filename"]
    if not os.path.exists(path):
        with st.spinner(f"Mengunduh {name}..."):
            r = requests.get(config["url"], stream=True)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success(f"{name} berhasil diunduh!")

@st.cache_resource
def load_model(path: str):
    try:
        return YOLO(path)
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        st.error(f"Error memuat model: {e}. File mungkin korup, silakan coba lagi.")
        return None

def run_detection(model, image: Image.Image):
    results   = model.predict(image, classes=[0, 1], conf=0.25)
    plotted   = results[0].plot()
    plotted   = plotted[:, :, ::-1]  

    boxes     = results[0].boxes
    conf_list = boxes.conf.tolist() if boxes is not None else []
    n_det     = len(conf_list)
    avg_conf  = float(np.mean(conf_list)) * 100 if conf_list else 0.0

    class_ids   = boxes.cls.tolist() if boxes is not None else []
    class_names = results[0].names
    labels      = [class_names[int(c)] for c in class_ids]

    return {
        "image"   : plotted,
        "n_det"   : n_det,
        "avg_conf": avg_conf,
        "labels"  : labels,
    }

def score_model(result: dict) -> float:
    return result["avg_conf"] * (1 + result["n_det"] * 0.05)

st.title("🛡️ Deteksi Hard Hat & Head")
st.caption("Upload gambar dan pilih model untuk mendeteksi pemakaian helm pekerja.")
st.divider()

all_model_names = list(MODEL_CONFIG.keys())
max_models      = len(all_model_names)

# ── Pilih jumlah model ──────────────────
st.subheader("Langkah 1 – Pilih jumlah model")

if max_models == 1:
    st.info("ℹ️ Saat ini hanya tersedia 1 model. Tambahkan `best2.pt` dan `best3.pt` ke repo untuk mengaktifkan perbandingan.")
    num_models = 1
else:
    num_options = list(range(1, max_models + 1))
    num_models  = st.radio(
        "Berapa model yang ingin digunakan?",
        options=num_options,
        format_func=lambda x: f"{x} model",
        horizontal=True,
    )

# ── Pilih model mana ────────────────────
st.subheader("Langkah 2 – Pilih model")

if num_models == 1:
    selected_names = [st.selectbox("Pilih satu model:", all_model_names)]
elif num_models == 2:
    col_a, col_b = st.columns(2)
    with col_a:
        m1 = st.selectbox("Model pertama:", all_model_names, index=0, key="m1")
    with col_b:
        remaining = [m for m in all_model_names if m != m1]
        m2 = st.selectbox("Model kedua:", remaining, key="m2")
    selected_names = [m1, m2]
else:
    selected_names = all_model_names
    st.info("Semua model akan dijalankan bersamaan.")

st.subheader("Langkah 3 – Upload gambar")
uploaded_file = st.file_uploader(
    "Pilih gambar (JPG / JPEG / PNG):",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=False, width=480)

    if st.button("Mulai Deteksi", type="primary"):
            models = {}
            with st.spinner("Memuat model..."):
                for name in selected_names:
                    cfg = MODEL_CONFIG[name]
                    # Langsung panggil load_model karena file sudah ada di folder
                    models[name] = load_model(cfg["filename"]) 
            
            results = {}
            with st.spinner("Memproses gambar..."):
                for name, mdl in models.items():
                    results[name] = run_detection(mdl, image)
        
        # ... sisa kode untuk menampilkan hasil ...
        st.divider()
        st.subheader("📊 Hasil Deteksi")

        if num_models == 1:
            name = selected_names[0]
            res  = results[name]
            st.markdown(f"#### {name}")
            st.image(res["image"], caption="Hasil Deteksi", use_column_width=True)
            c1, c2 = st.columns(2)
            c1.metric("Jumlah Deteksi", res["n_det"])
            c2.metric("Rata-rata Confidence", f"{res['avg_conf']:.1f}%")

        else:
            scores    = {name: score_model(results[name]) for name in selected_names}
            best_name = max(scores, key=scores.get)
            cols      = st.columns(len(selected_names))

            for col, name in zip(cols, selected_names):
                res     = results[name]
                is_best = (name == best_name)
                with col:
                    badge = " 🏆 **Terbaik!**" if is_best else ""
                    st.markdown(f"##### {name}{badge}")
                    st.image(res["image"], use_column_width=True)

                    c1, c2 = st.columns(2)
                    c1.metric("Deteksi", res["n_det"])
                    c2.metric("Conf (%)", f"{res['avg_conf']:.1f}")

                    if res["labels"]:
                        st.caption("Label: " + ", ".join(set(res["labels"])))
                    else:
                        st.caption("Tidak ada objek terdeteksi.")

                    if is_best:
                        st.success("Model ini menghasilkan deteksi terbaik.")

            st.divider()
            best_res = results[best_name]
            st.markdown(
                f"### 🏆 Model Terbaik: **{best_name}**\n"
                f"Confidence rata-rata **{best_res['avg_conf']:.1f}%** "
                f"dengan **{best_res['n_det']}** objek terdeteksi."
            )

        st.success("✅ Deteksi selesai!")
