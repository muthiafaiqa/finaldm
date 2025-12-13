import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(
    page_title="Sistem Cerdas Toko Bangunan",
    page_icon="🏗️",
    layout="wide"
)

# ==================================================
# CSS
# ==================================================
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #2c3e50; }
[data-testid="stSidebar"] * { color: white !important; }
h1 { color: #e67e22; font-weight: 800; text-align: center; }
.card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
}
.stButton>button {
    background: linear-gradient(135deg, #e67e22, #d35400);
    color: white;
    font-weight: bold;
    width: 100%;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# PATH AMAN (CLOUD SAFE)
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "TRANSAKSI_PENJUALAN_PRODUK_TOKO_BANGUNAN_SYNTHETIC.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model_rf_harga.pkl")

# ==================================================
# LOAD DATA & MODEL
# ==================================================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    model, features = joblib.load(MODEL_PATH)
    return model, features

df = load_data()
model, feature_columns = load_model()

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("🏗️ Navigasi")
menu = st.sidebar.radio(
    "Menu",
    ["🏠 Home", "💰 Prediksi Harga", "📊 Segmentasi Pelanggan"]
)

# ==================================================
# HOME
# ==================================================
if menu == "🏠 Home":
    st.title("🏗️ Sistem Cerdas Toko Bangunan")
    st.markdown("<div class='card'>Aplikasi Data Mining berbasis Machine Learning.</div>", unsafe_allow_html=True)

# ==================================================
# REGRESI
# ==================================================
elif menu == "💰 Prediksi Harga":
    st.title("💰 Prediksi Total Harga")

    col1, col2 = st.columns(2)
    with col1:
        qty = st.number_input("Jumlah Barang", min_value=1, value=5)
        harga = st.number_input("Harga Satuan (Rp)", min_value=1000, value=50000, step=1000)
    with col2:
        kategori = st.selectbox("Kategori", ["Alat", "Bahan Logam dan PVC", "Cat", "Material Konstruksi"])

    if st.button("HITUNG"):
        input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
        input_df["Harga Satuan"] = harga
        input_df["Kuantitas"] = qty
        input_df[f"Kategori_{kategori}"] = 1

        pred = model.predict(input_df)[0]
        st.success(f"💵 Estimasi Total: Rp {pred:,.0f}")
        st.balloons()

# ==================================================
# CLUSTERING
# ==================================================
elif menu == "📊 Segmentasi Pelanggan":
    st.title("👥 Segmentasi Pelanggan")

    k = st.slider("Jumlah Cluster", 2, 5, 3)

    df_cluster = df.groupby("ID Transaksi").agg({
        "Total Harga": "sum",
        "Kuantitas": "sum"
    }).reset_index()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_cluster[["Total Harga", "Kuantitas"]])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_cluster["Cluster"] = kmeans.fit_predict(scaled)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df_cluster,
        x="Total Harga",
        y="Kuantitas",
        hue="Cluster",
        palette="viridis",
        s=120,
        ax=ax
    )
    st.pyplot(fig)

    st.dataframe(df_cluster.head(), use_container_width=True)
