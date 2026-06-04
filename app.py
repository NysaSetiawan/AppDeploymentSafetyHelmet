import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import requests
import numpy as np

# ─────────────────────────────────────────
# 1. Konfigurasi Halaman
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Helm Proyek",
    page_icon="🛡️",
    layout="wide"
)

# ─────────────────────────────────────────
# 2. Konfigurasi Model
#    Ganti nama & URL sesuai file best.pt Anda
# ─────────────────────────────────────────
MODEL_CONFIG = {
    "Model 1 – YOLOv8n": {
        "filename": "best1.pt",
        "url": "URL_DOWNLOAD_MODEL_1",      # ← Ganti dengan URL publik Anda
        "description": "Ringan & cepat",
    },
    "Model 2 – YOLOv8s": {
        "filename": "best2.pt",
        "url": "URL_DOWNLOAD_MODEL_2",      # ← Ganti dengan URL publik Anda
        "description": "Seimbang",
    },
    "Model 3 – YOLOv8m": {
        "filename": "best3.pt",
        "url": "URL_DOWNLOAD_MODEL_3",      # ← Ganti dengan URL publik Anda
        "description": "Akurasi tinggi",
    },
}

# ─────────────────────────────────────────
# 3. Fungsi Download Model
# ─────────────────────────────────────────
def download_model(name: str, config: dict):
    path = config["filename"]
    if not os.path.exists(path):
        with st.spinner(f"⏬ Mengunduh {name}..."):
            r = requests.get(config["url"], stream=True)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success(f"✅ {name} berhasil diunduh!")

# ─────────────────────────────────────────
# 4. Load Model (di-cache agar tidak reload)
# ─────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    return YOLO(path)

# ─────────────────────────────────────────
# 5. Fungsi Deteksi & Hitung Skor
# ─────────────────────────────────────────
def run_detection(model, image: Image.Image):
    results = model.predict(image, classes=[0, 1], conf=0.25)
    plotted  = results[0].plot()           # numpy array BGR
    plotted  = plotted[:, :, ::-1]         # BGR → RGB

    boxes      = results[0].boxes
    conf_list  = boxes.conf.tolist() if boxes is not None else []
    n_det      = len(conf_list)
    avg_conf   = float(np.mean(conf_list)) * 100 if conf_list else 0.0

    # Label per kelas
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
    """Skor sederhana: rata-rata confidence × jumlah deteksi (bisa dikustomisasi)."""
    return result["avg_conf"] * (1 + result["n_det"] * 0.05)

# ─────────────────────────────────────────
# 6. Tampilan Utama
# ─────────────────────────────────────────
st.title("🛡️ Deteksi Hard Hat & Head")
st.caption("Upload gambar dan pilih model untuk mendeteksi pemakaian helm pekerja.")

st.divider()

# ── Pilih jumlah model ──────────────────
st.subheader("Langkah 1 – Pilih jumlah model")
num_models = st.radio(
    "Berapa model yang ingin digunakan?",
    options=[1, 2, 3],
    format_func=lambda x: f"{'Satu' if x==1 else 'Dua' if x==2 else 'Tiga'} model ({x})",
    horizontal=True,
)

all_model_names = list(MODEL_CONFIG.keys())

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
else:  # 3 model
    selected_names = all_model_names  # langsung pakai ketiganya
    st.info("Semua tiga model akan dijalankan bersamaan.")

# ── Upload Gambar ───────────────────────
st.subheader("Langkah 3 – Upload gambar")
uploaded_file = st.file_uploader(
    "Pilih gambar (JPG / JPEG / PNG):",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=False, width=480)

    if st.button("🚀 Mulai Deteksi", type="primary"):
        # Download & load semua model yang dipilih
        models = {}
        for name in selected_names:
            cfg = MODEL_CONFIG[name]
            download_model(name, cfg)
            models[name] = load_model(cfg["filename"])

        # Jalankan deteksi
        results = {}
        with st.spinner("Memproses gambar..."):
            for name, mdl in models.items():
                results[name] = run_detection(mdl, image)

        st.divider()
        st.subheader("📊 Hasil Deteksi")

        # ── Tampilan: 1 model ───────────────────
        if num_models == 1:
            name   = selected_names[0]
            res    = results[name]
            st.markdown(f"#### {name}")
            st.image(res["image"], caption="Hasil Deteksi", use_column_width=True)
            st.metric("Jumlah Deteksi", res["n_det"])
            st.metric("Rata-rata Confidence", f"{res['avg_conf']:.1f}%")

        # ── Tampilan: 2 atau 3 model (side-by-side) ─
        else:
            cols = st.columns(len(selected_names))
            scores = {name: score_model(results[name]) for name in selected_names}
            best_name = max(scores, key=scores.get)

            for col, name in zip(cols, selected_names):
                res = results[name]
                is_best = (name == best_name)
                with col:
                    badge = " 🏆 **Terbaik!**" if is_best else ""
                    st.markdown(f"##### {name}{badge}")
                    st.image(res["image"], use_column_width=True)

                    # Metrik ringkas
                    c1, c2 = st.columns(2)
                    c1.metric("Deteksi", res["n_det"])
                    c2.metric("Conf (%)", f"{res['avg_conf']:.1f}")

                    # Label yang terdeteksi
                    if res["labels"]:
                        unique_labels = set(res["labels"])
                        st.caption("Label: " + ", ".join(unique_labels))
                    else:
                        st.caption("Tidak ada objek terdeteksi.")

                    if is_best:
                        st.success("Model ini menghasilkan deteksi terbaik.")

            # ── Ringkasan pemenang ─────────────────
            st.divider()
            best_res = results[best_name]
            st.markdown(
                f"### 🏆 Model Terbaik: **{best_name}**\n"
                f"Confidence rata-rata **{best_res['avg_conf']:.1f}%** "
                f"dengan **{best_res['n_det']}** objek terdeteksi — "
                f"skor tertinggi di antara model yang dipilih."
            )

        st.success("✅ Deteksi selesai!")
