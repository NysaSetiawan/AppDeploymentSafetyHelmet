import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import requests

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Helm Proyek", layout="centered")

# 2. Fungsi Download Model dari Google Drive (Ganti dengan link publik Anda)
def download_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.warning("Model sedang diunduh, harap tunggu...")
        # GANTI URL INI dengan link download publik file best.pt Anda
        url = "URL_DOWNLOAD_GOOGLE_DRIVE_ANDA"
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
        st.success("Model berhasil diunduh!")

# 3. Load Model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

# Jalankan download jika belum ada
download_model()
model = load_model()

# 4. Tampilan Antarmuka
st.title("🛡️ Aplikasi Deteksi Hard Hat & Head")
st.write("Upload gambar untuk mendeteksi apakah pekerja menggunakan helm.")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    
    if st.button("Mulai Deteksi"):
        with st.spinner("Sedang memproses..."):
            # Prediksi dengan filter class 0 (head) dan 1 (helmet)
            results = model.predict(image, classes=[0, 1], conf=0.25)
            
            # Plot hasil
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="Hasil Deteksi", use_column_width=True)
            st.success("Deteksi selesai!")