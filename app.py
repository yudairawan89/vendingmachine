# ======================================
# === app.py: Vending Machine GUI ===
# ======================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

# ======================================
# === STEP 1: Load Artefak Model ===
# ======================================

# Load base models
rf = joblib.load("RandomForest_model.pkl")
svm = joblib.load("SVM_model.pkl")
xgb = joblib.load("XGBoost_model.pkl")

# Load meta LSTM (tanpa compile untuk menghindari error)
lstm_meta = tf.keras.models.load_model("stacking_lstm_meta.h5", compile=False)

# Load scaler & encoder
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Link Google Sheets (CSV export)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1AvzsaiDZqQ_0tR7S3_OYCqwIeTIz3GQ27ptvT4GWk6A/export?format=csv"


# ======================================
# === STEP 2: Preprocessing Fungsi ===
# ======================================
def preprocess_data(df):
    """Preprocessing data sesuai training pipeline"""
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df = df.groupby(['date', 'sku']).agg(
        total_sold=('sold', 'sum'),
        avg_price=('price', 'mean')
    ).reset_index()

    # Tambah fitur waktu
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Urutkan data
    df = df.sort_values(['sku', 'date'])

    # Tambah lag features
    for lag in [1, 2, 3, 7, 14]:
        df[f'lag_{lag}'] = df.groupby('sku')['total_sold'].shift(lag)

    # Tambah rolling mean & std
    for win in [3, 7, 14]:
        df[f'rolling_mean_{win}'] = (
            df.groupby('sku')['total_sold'].shift(1).rolling(window=win, min_periods=1).mean()
        )
        df[f'rolling_std_{win}'] = (
            df.groupby('sku')['total_sold'].shift(1).rolling(window=win, min_periods=1).std()
        )

    # Total demand harian & share
    df['total_demand_day'] = df.groupby('date')['total_sold'].transform('sum')
    df['sku_share'] = df['total_sold'] / (df['total_demand_day'] + 1e-5)

    # Encode SKU
    df['sku_encoded'] = le.transform(df['sku'])

    return df


def prepare_features(latest_row):
    """Siapkan 1 baris fitur untuk prediksi"""
    features = latest_row.drop(['date', 'sku', 'total_sold'], errors='ignore')
    return features.to_frame().T


def predict_sales(input_df):
    """Prediksi penjualan dengan stacking"""
    X_scaled = scaler.transform(input_df)

    # Prediksi base models
    base_preds = [
        rf.predict(X_scaled).reshape(-1, 1),
        svm.predict(X_scaled).reshape(-1, 1),
        xgb.predict(X_scaled).reshape(-1, 1)
    ]
    meta_X = np.hstack(base_preds)
    meta_X_lstm = meta_X.reshape((meta_X.shape[0], 1, meta_X.shape[1]))

    # Prediksi meta LSTM
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
        raw_df = pd.read_csv(SHEET_URL)
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'], errors='coerce')

        # Ambil data terakhir untuk monitoring suhu & kelembaban
        latest_raw = raw_df.sort_values("timestamp").iloc[-1]

        col1, col2, col3 = st.columns(3)
        col1.metric("🌡️ Suhu (°C)", f"{latest_raw['temperature_c']}")
        col2.metric("💧 Kelembaban (%)", f"{latest_raw['humidity']}")
        col3.metric("🥤 SKU", latest_raw['sku'])

        st.dataframe(raw_df.tail(10))

        # Preprocessing
        processed = preprocess_data(raw_df)
        latest_processed = processed.dropna().iloc[-1]

        # Siapkan fitur untuk prediksi
        features = prepare_features(latest_processed)
        pred = predict_sales(features)

        st.success(f"🔮 Prediksi Penjualan SKU {latest_raw['sku']}: {pred[0]:.2f} unit")

    except Exception as e:
        st.error(f"Gagal mengambil data Google Sheet: {e}")


# -------------------------------
# Tab 2: Manual Input
# -------------------------------
with tab2:
    st.subheader("Input Manual untuk Uji Coba Prediksi")

    sku = st.text_input("SKU (nama produk)", "Nescafe")
    avg_price = st.number_input("Harga Rata-rata", min_value=1000, max_value=20000, value=10000, step=500)
    day_of_week = st.selectbox("Hari ke-", list(range(7)),
                               format_func=lambda x: ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"][x])
    is_weekend = 1 if day_of_week >= 5 else 0

    # Encode SKU
    try:
        sku_encoded = le.transform([sku])[0]
    except:
        sku_encoded = -1  # fallback jika SKU baru

    if st.button("Prediksi Penjualan"):
        input_manual = pd.DataFrame([{
            "avg_price": avg_price,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "sku_encoded": sku_encoded,
            # Fitur tambahan default (bisa diisi manual juga)
            "lag_1": 0, "lag_2": 0, "lag_3": 0, "lag_7": 0, "lag_14": 0,
            "rolling_mean_3": 0, "rolling_mean_7": 0, "rolling_mean_14": 0,
            "rolling_std_3": 0, "rolling_std_7": 0, "rolling_std_14": 0,
            "total_demand_day": 0, "sku_share": 0
        }])

        pred = predict_sales(input_manual)
        st.success(f"🔮 Prediksi Penjualan (Manual): {pred[0]:.2f} unit")
