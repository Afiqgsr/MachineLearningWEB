import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random

# Load data dan model
df = pd.read_csv('CarPrice.csv')
model = pickle.load(open('model_prediksi_harga_mobil.sav', 'rb'))

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Harga Mobil", layout="wide")

# Tab utama
tab_home, tab_prediksi = st.tabs(["🏠 Beranda", "🚗 Prediksi Harga"])

# === Tab Beranda ===
with tab_home:
    st.title("🚘 Prediksi Harga Mobil dengan Machine Learning")
    st.markdown("Aplikasi ini memprediksi harga mobil berdasarkan fitur-fitur seperti horsepower, curb weight, engine size, dan highway MPG menggunakan model **Linear Regression**.")

    # Dataset
    st.subheader("📊 Dataset Mobil")
    st.dataframe(df, use_container_width=True)

    # Statistik dasar
    st.markdown("### 📈 Statistik Harga")
    col1, col2, col3 = st.columns(3)
    col1.metric("Harga Tertinggi", f"${df['price'].max():,.0f}")
    col2.metric("Harga Terendah", f"${df['price'].min():,.0f}")
    col3.metric("Rata-rata Harga", f"${df['price'].mean():,.0f}")

    # Visualisasi
    st.markdown("---")
    st.subheader("📉 Visualisasi Fitur Mobil")
    tab1, tab2, tab3, tab4 = st.tabs(["Horsepower", "Curb Weight", "Engine Size", "Highway MPG"])
    with tab1:
        st.area_chart(df['horsepower'])
    with tab2:
        st.line_chart(df['curbweight'])
    with tab3:
        st.bar_chart(df['enginesize'])
    with tab4:
        st.line_chart(df['highwaympg'])

# === Tab Prediksi ===
with tab_prediksi:
    st.title("🧮 Prediksi Harga Mobil")

    st.sidebar.header("🔧 Input Fitur Mobil")
    horsepower = st.sidebar.slider("Horsepower", int(df['horsepower'].min()), int(df['horsepower'].max()), int(df['horsepower'].mean()))
    curbweight = st.sidebar.slider("Curb Weight", int(df['curbweight'].min()), int(df['curbweight'].max()), int(df['curbweight'].mean()))
    enginesize = st.sidebar.slider("Engine Size", int(df['enginesize'].min()), int(df['enginesize'].max()), int(df['enginesize'].mean()))
    highwaympg = st.sidebar.slider("Highway MPG", int(df['highwaympg'].min()), int(df['highwaympg'].max()), int(df['highwaympg'].mean()))

    st.markdown("Silakan klik tombol di bawah untuk memprediksi harga mobil berdasarkan input.")

    if st.button("🔍 Prediksi Harga Mobil"):
        input_data = np.array([[horsepower, curbweight, enginesize, highwaympg]])
        prediksi = model.predict(input_data)[0]
        st.subheader("📌 Hasil Prediksi")
        st.success(f"💰 Harga mobil yang diprediksi adalah: **${prediksi:,.2f}**")
        st.balloons()

        motivasi = random.choice([
            "Tetap semangat mencari mobil impian! 🚗💨",
            "Data membawa keputusan cerdas! 📊",
            "Jangan lupa cek fitur lainnya sebelum membeli. 😉",
            "Terus eksplorasi dan bandingkan harga terbaik! 🔍"
        ])
        st.info(motivasi)
