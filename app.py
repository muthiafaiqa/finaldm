import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# SETUP HALAMAN
st.set_page_config(page_title="Sistem Cerdas Toko Bangunan", layout="wide")

# PATH FILES
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "TRANSAKSI_PENJUALAN_PRODUK_TOKO_BANGUNAN_SYNTHETIC.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model_rf_harga.pkl")

# LOAD MODEL & DATA
@st.cache_resource
def load_stuff():
    # Load Model
    try:
        model, features = joblib.load(MODEL_PATH)
    except:
        return None, None, None
    
    # Load Data untuk Clustering
    try:
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.replace(' ', '_') # Samakan format
    except:
        return model, features, None
        
    return model, features, df

model, feature_columns, df = load_stuff()

# CEK ERROR AWAL
if model is None:
    st.error("File 'model_rf_harga.pkl' tidak ditemukan atau rusak. Silakan upload ulang di GitHub.")
    st.stop()
if df is None:
    st.error("File CSV data tidak ditemukan.")
    st.stop()

# NAVIGASI
menu = st.sidebar.radio("Menu", ["🏠 Home", "💰 Prediksi Harga", "📊 Segmentasi Pelanggan"])

# MENU 1: HOME
if menu == "🏠 Home":
    st.title("Sistem Toko Bangunan")
    st.write("Selamat Datang di Aplikasi Prediksi & Segmentasi.")

# MENU 2: PREDIKSI (REGRESI)
elif menu == "💰 Prediksi Harga":
    st.title("💰 Prediksi Total Harga")
    
    col1, col2 = st.columns(2)
    with col1:
        qty = st.number_input("Jumlah", min_value=1, value=10)
        harga = st.number_input("Harga Satuan (Rp)", min_value=100, value=50000)
    with col2:
        # Nama kategori persis seperti di CSV asli (pakai spasi gapapa)
        kat = st.selectbox("Kategori", ["Alat", "Bahan Logam dan PVC", "Cat", "Material Konstruksi"])

    if st.button("Hitung"):
        # 1. Siapkan wadah data (semua 0)
        input_data = {col: [0] for col in feature_columns}
        
        # 2. Isi data Angka (Cari kolom yang pas)
        if 'Harga_Satuan' in feature_columns: input_data['Harga_Satuan'] = [harga]
        elif 'Harga Satuan' in feature_columns: input_data['Harga Satuan'] = [harga]
        
        input_data['Kuantitas'] = [qty]
        
        # 3. Isi data Kategori (Ganti spasi jadi underscore biar cocok sama model)
        # Misal: "Bahan Logam" -> "Kategori_Bahan_Logam..."
        kat_bersih = kat.replace(" ", "_")
        nama_kolom_kategori = f"Kategori_{kat_bersih}"
        
        if nama_kolom_kategori in feature_columns:
            input_data[nama_kolom_kategori] = [1]
        
        # 4. Prediksi
        input_df = pd.DataFrame(input_data)
        input_df = input_df[feature_columns] # Urutkan sesuai kemauan model
        
        hasil = model.predict(input_df)[0]
        st.success(f"Estimasi: Rp {hasil:,.0f}")

# MENU 3: CLUSTERING
elif menu == "📊 Segmentasi Pelanggan":
    st.title("Segmentasi Pelanggan")
    k = st.slider("Jumlah Kelompok", 2, 5, 3)
    
    if st.button("Mulai Clustering"):
        # Agregasi Data
        df_group = df.groupby("ID_Transaksi").agg({
            "Total_Harga": "sum", "Kuantitas": "sum"
        }).reset_index()
        
        # Scaling & KMeans
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_group[["Total_Harga", "Kuantitas"]])
        
        km = KMeans(n_clusters=k, n_init=10)
        df_group["Cluster"] = km.fit_predict(scaled)
        
        # Plot
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_group, x="Total_Harga", y="Kuantitas", hue="Cluster", palette="viridis", ax=ax)
        st.pyplot(fig)
