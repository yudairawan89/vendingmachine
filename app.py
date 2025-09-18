import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ======================================
# === STEP 1: Konfigurasi & Load Model ===
# ======================================

# Load base models
rf = joblib.load("saved_models/RandomForest_model.pkl")
svm = joblib.load("saved_models/SVM_model.pkl")
xgb = joblib.load("saved_models/XGBoost_model.pkl")

# Load meta LSTM
lstm_meta = tf.keras.models.load_model("saved_models/stacking_lstm_meta.h5")

# Load scaler
scaler = joblib.load("saved_models/scaler.pkl")  # pastikan Anda sudah save scaler saat training

# Link Google Sheets (gunakan CSV export)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1AvzsaiDZqQ_0tR7S3_OYCqwIeTIz3GQ27ptvT4GWk6A/export?format=csv"


# ======================================
# === STEP 2: Fungsi Prediksi ===
# ======================================
def predict_sales(input_df):
    X_scaled = scaler.transform(input_df)

    # Base model predictions
    base_preds = [
        rf.predict(X_scaled).reshape(-1, 1),
        svm.predict(X_scaled).reshape(-1, 1),
        xgb.predict(X_scaled).reshape(-1, 1)
    ]
    meta_X = np.hstack(base_preds)
    meta_X_lstm = meta_X.reshape((meta_X.shape[0], 1, meta_X.shape[1]))

    # Meta prediction
    y_pred = lstm_meta.predict(meta_X_lstm, verbose=0)
    return y_pred.flatten()


# ======================================
# === STEP 3: Streamlit GUI ===
# ======================================
st.set_page_config(page_title="Vending Machine Prediction", layout="wide")
st.title("🤖 Vending Machine Monitoring & Prediction")

# Tabs
tab1, tab2 = st.tabs(["📡 Monitoring", "📝 Manual Input"])

# -------------------------------
# Tab 1: Monitoring (Google Sheet)
# -------------------------------
with tab1:
    st.subheader("Data Monitoring dari Google Sheet")

    try:
        df = pd.read_csv(SHEET_URL)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        latest = df.sort_values("timestamp").iloc[-1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Suhu (°C)", f"{latest['temperature_c']}")
        col2.metric("Kelembaban (%)", f"{latest['humidity']}")
        col3.metric("SKU", latest['sku'])

        st.dataframe(df.tail(10))

        # Siapkan data untuk prediksi (pakai kolom yang sama dengan training)
        features = pd.DataFrame([{
            "avg_price": latest["price"],
            "day_of_week": pd.to_datetime(latest["timestamp"]).dayofweek,
            "is_weekend": 1 if pd.to_datetime(latest["timestamp"]).dayofweek >= 5 else 0,
            "sku_encoded": 0  # sementara 0, bisa dihubungkan dengan LabelEncoder saat training
        }])

        pred = predict_sales(features)
        st.success(f"🔮 Prediksi Penjualan: {pred[0]:.2f} unit")

    except Exception as e:
        st.error(f"Gagal mengambil data Google Sheet: {e}")


# -------------------------------
# Tab 2: Manual Input
# -------------------------------
with tab2:
    st.subheader("Input Manual untuk Uji Coba Prediksi")

    sku = st.text_input("SKU (nama produk)", "Nescafe")
    avg_price = st.number_input("Harga Rata-rata", min_value=1000, max_value=20000, value=10000, step=500)
    day_of_week = st.selectbox("Hari ke-", list(range(7)), format_func=lambda x: ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"][x])
    is_weekend = 1 if day_of_week >= 5 else 0
    sku_encoded = 0  # untuk sekarang dummy, bisa mapping LabelEncoder

    if st.button("Prediksi Penjualan"):
        input_manual = pd.DataFrame([{
            "avg_price": avg_price,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "sku_encoded": sku_encoded
        }])
        pred = predict_sales(input_manual)
        st.success(f"🔮 Prediksi Penjualan (Manual): {pred[0]:.2f} unit")
