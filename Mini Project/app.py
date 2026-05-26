import streamlit as str
import pandas as pd
import numpy as np
from utils.data_preprocessing import generate_dummy_data, preprocess_features
from src.model_pipeline import train_and_evaluate, plot_confusion_matrix

str.set_page_config(page_title="AI Performa Akademik", layout="wide")
str.title("🎓 Sistem Analisis Prediktif Performa Akademik Mahasiswa")

@str.cache_resource
def kompilasi_sistem_ai():
    df_raw = generate_dummy_data(n_samples=600)
    X_scaled, scaler_terlatih = preprocess_features(df_raw, is_training=True)
    model_lr, model_knn, X_test, y_test = train_and_evaluate(df_raw, X_scaled)
    return model_lr, model_knn, scaler_terlatih, X_test, y_test

model_lr, model_knn, scaler, X_test, y_test = kompilasi_sistem_ai()

kolom_input, kolom_hasil = str.columns([1, 2])

with kolom_input:
    str.header("📥 Input Profil Mahasiswa")
    ipk = str.slider("Indeks Prestasi Kumulatif (IPK) Sem 1:", 0.0, 4.0, 3.0, step=0.05)
    kehadiran = str.slider("Persentase Kehadiran Kelas (%):", 0, 100, 85, step=1)
    jam_belajar = str.number_input("Rata-rata Jam Belajar Mandiri / Hari:", min_value=0.0, max_value=24.0, value=3.5, step=0.5)
    pilihan_model = str.selectbox("Pilih Algoritma AI:", ["Logistic Regression", "K-Nearest Neighbors (KNN)"])

with kolom_hasil:
    str.header("📊 Hasil Analisis Prediksi AI")
    data_baru = pd.DataFrame([[ipk, kehadiran, jam_belajar]], columns=['ipk_sem1', 'kehadiran', 'jam_belajar_harian'])
    data_baru_scaled = preprocess_features(data_baru, is_training=False, scaler=scaler)
    
    if pilihan_model == "Logistic Regression":
        prediksi = model_lr.predict(data_baru_scaled)[0]
        probabilitas = model_lr.predict_proba(data_baru_scaled)[0][1] * 100
    else:
        prediksi = model_knn.predict(data_baru_scaled)[0]
        probabilitas = model_knn.predict_proba(data_baru_scaled)[0][1] * 100

    if prediksi == 1:
        str.success(f"### 🎉 Hasil: DIPREDIKSI LULUS TEPAT WAKTU")
        str.write(f"Tingkat keyakinan: **{probabilitas:.2f}%**")
    else:
        str.error(f"### ⚠️ Hasil: BERISIKO TERLAMBAT / DROP-OUT")
        str.write(f"Tingkat risiko: **{100 - probabilitas:.2f}%**")
        str.warning("💡 **Rekomendasi:** Perlu bimbingan konseling akademik khusus.")

    str.markdown("---")
    str.subheader("🎯 Metrik Validasi Kecerdasan Model")
    kolom_grafik1, kolom_grafik2 = str.columns(2)
    with kolom_grafik1:
        fig_lr = plot_confusion_matrix(y_test, model_lr.predict(X_test), title="Confusion Matrix - Logistic Regression")
        str.pyplot(fig_lr)
    with kolom_grafik2:
        fig_knn = plot_confusion_matrix(y_test, model_knn.predict(X_test), title="Confusion Matrix - KNN")
        str.pyplot(fig_knn)
