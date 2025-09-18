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

# ======================================
# === STEP 1: Konfigurasi & Load Model ===
# ======================================

# Load base models
rf = joblib.load("RandomForest_model.pkl")
svm = joblib.load("SVM_model.pkl")
xgb = joblib.load("XGBoost_model.pkl")

# Load meta LSTM
lstm_meta = tf.keras.models.load_model("stacking_lstm_meta.h5", compile=False)

# Load scaler & LabelEncoder
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Link Google Sheets (gunakan CSV export)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1AvzsaiDZqQ_0tR7S3_OYCqwIeTIz3GQ27ptvT4GWk6A/export?format=csv"


# ======================================
# === STEP 2: Fungsi Prediksi ===
# ======================================
def predict_sales(input_df):
    X_scaled = scaler.transform(input_df)

    base_preds = [
        rf.predict(X_scaled).reshape(-1, 1),
        svm.predict(X_scaled).reshape(-1, 1),
        xgb.predict(X_scaled).reshape(-1, 1)
    ]
    meta_X = np.hstack(base_preds)
    meta_X_lstm = meta_X.reshape((meta_X.shape[0], 1, meta_X.shape[1]))

    y_pred = lstm_meta.predict(meta_X_lstm, verbose=0)
    return y_pred.flatten()


# ======================================
# === STEP 3: Preprocessing ===
# ======================================
def preprocess_data(df):
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_sales = df.groupby(['date', 'sku']).agg(
        total_sold=('sold', 'sum'),
        avg_price=('price', 'mean')
    ).reset_index()

    daily_sales['date'] = pd.to_datetime(daily_sales['date'])
    daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek
    daily_sales['is_weekend'] = (daily_sales['day_of_week'] >= 5).astype(int)
    daily_sales = daily_sales.sort_values(['sku', 'date'])

    for lag in [1, 2, 3, 7, 14]:
        daily_sales[f'lag_{lag}'] = daily_sales.groupby('sku')['total_sold'].shift(lag)

    for win in [3, 7, 14]:
        daily_sales[f'rolling_mean_{win}'] = (
            daily_sales.groupby('sku')['total_sold']
            .shift(1).rolling(window=win, min_periods=1).mean()
        )
        daily_sales[f'rolling_std_{win}'] = (
            daily_sales.groupby('sku')['total_sold']
            .shift(1).rolling(window=win, min_periods=1).std()
        )

    daily_sales['total_demand_day'] = daily_sales.groupby('date')['total_sold'].transform('sum')
    daily_sales['sku_share'] = daily_sales['total_sold'] / (daily_sales['total_demand_day'] + 1e-5)
    daily_sales['sku_encoded'] = le.transform(daily_sales['sku'])

    return daily_sales


def prepare_features(row):
    return pd.DataFrame([row.drop(labels=['date', 'sku', 'total_sold'])])


# ======================================
# === STEP 4: Streamlit GUI ===
# ======================================
st.set_page_config(page_title="Vending Machine Prediction", layout="wide")
st.title("🤖 Vending Machine Monitoring & Prediction")

tab1, tab2 = st.tabs(["📡 Monitoring", "📝 Manual Input"])

# -------------------------------
# Tab 1: Monitoring
# -------------------------------
with tab1:
    st.subheader("Data Monitoring dari Google Sheet")

    try:
        df = pd.read_csv(SHEET_URL)

        if df.empty:
            st.warning("⚠️ Google Sheet kosong, belum ada data masuk.")
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])

            if df.empty:
                st.warning("⚠️ Data ada, tapi kolom timestamp kosong/tidak valid.")
            else:
                latest = df.sort_values("timestamp").iloc[-1]

                col1, col2, col3 = st.columns(3)
                col1.metric("🌡️ Suhu (°C)", f"{latest['temperature_c']}")
                col2.metric("💧 Kelembaban (%)", f"{latest['humidity']}")
                col3.metric("🥤 SKU", latest['sku'])

                st.dataframe(df.tail(10))

                processed = preprocess_data(df)
                if processed.dropna().empty:
                    st.warning("⚠️ Data tidak cukup untuk preprocessing (lag/rolling).")
                else:
                    latest_processed = processed.dropna().iloc[-1]
                    features = prepare_features(latest_processed)
                    pred = predict_sales(features)
                    st.success(f"🔮 Prediksi Penjualan SKU {latest['sku']}: {pred[0]:.2f} unit")

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
                               format_func=lambda x: ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][x])
    is_weekend = 1 if day_of_week >= 5 else 0

    try:
        sku_encoded = le.transform([sku])[0]
    except:
        sku_encoded = 0
        st.warning(f"SKU '{sku}' tidak ada di encoder, diset ke 0.")

    if st.button("Prediksi Penjualan"):
        input_manual = pd.DataFrame([{
            "avg_price": avg_price,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "sku_encoded": sku_encoded
        }])
        pred = predict_sales(input_manual)
        st.success(f"🔮 Prediksi Penjualan (Manual): {pred[0]:.2f} unit")
