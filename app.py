# ======================================
# === app.py: Vending Machine GUI ===
# ======================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# ======================================
# === STEP 1: Konfigurasi & Load Model ===
# ======================================

# Load base models
rf = joblib.load("RandomForest_model.pkl")
svm = joblib.load("SVM_model.pkl")
xgb = joblib.load("XGBoost_model.pkl")

# Load meta LSTM (tanpa compile)
lstm_meta = tf.keras.models.load_model("stacking_lstm_meta.h5", compile=False)

# Load scaler & encoder & feature names
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

# Link Google Sheets (gunakan CSV export)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1AvzsaiDZqQ_0tR7S3_OYCqwIeTIz3GQ27ptvT4GWk6A/export?format=csv"


# ======================================
# === STEP 2: Fungsi Prediksi ===
# ======================================
def predict_sales(input_df):
    # Pastikan kolom sama urutannya
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

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
# === STEP 3: Streamlit GUI ===
# ======================================
st.set_page_config(page_title="Vending Machine Prediction", layout="wide")
st.title("🤖 Vending Machine Monitoring & Prediction")

tab1, tab2, tab3 = st.tabs(["📡 Monitoring", "📝 Manual Input", "📅 Input Hari Ini → Prediksi Besok"])


# -------------------------------
# Tab 1: Monitoring
# -------------------------------
with tab1:
    st.subheader("Data Monitoring dari Google Sheet")

    try:
        df = pd.read_csv(SHEET_URL)

        # Parsing timestamp (ada yg pakai titik di jam)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=False)

        if df["timestamp"].isnull().all():
            st.warning("⚠️ Data ada, tapi kolom timestamp kosong/tidak valid.")
        else:
            latest = df.sort_values("timestamp").iloc[-1]

            col1, col2, col3 = st.columns(3)
            col1.metric("Suhu (°C)", f"{latest['temperature_c']}")
            col2.metric("Kelembaban (%)", f"{latest['humidity']}")
            col3.metric("SKU", latest['sku'])

            st.dataframe(df.tail(10))

            # Mapping SKU pakai LabelEncoder
            try:
                sku_encoded = le.transform([latest["sku"]])[0]
            except:
                sku_encoded = 0

            features = pd.DataFrame([{
                "avg_price": latest["price"],
                "day_of_week": latest["timestamp"].dayofweek,
                "is_weekend": 1 if latest["timestamp"].dayofweek >= 5 else 0,
                "sku_encoded": sku_encoded
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
    day_of_week = st.selectbox("Hari ke-", list(range(7)),
                               format_func=lambda x: ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"][x])
    is_weekend = 1 if day_of_week >= 5 else 0

    try:
        sku_encoded = le.transform([sku])[0]
    except:
        sku_encoded = 0

    if st.button("Prediksi Penjualan (Manual)"):
        input_manual = pd.DataFrame([{
            "avg_price": avg_price,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "sku_encoded": sku_encoded
        }])
        pred = predict_sales(input_manual)
        st.success(f"🔮 Prediksi Penjualan (Manual): {pred[0]:.2f} unit")


# -------------------------------
# Tab 3: Input Hari Ini → Prediksi Besok
# -------------------------------
with tab3:
    st.subheader("Input Data Hari Ini untuk Prediksi Besok")

    today = st.date_input("Tanggal Hari Ini")
    sku = st.selectbox("SKU Hari Ini", le.classes_)
    price = st.number_input("Harga Hari Ini", min_value=1000, max_value=20000, value=10000)
    sold = st.number_input("Jumlah Terjual Hari Ini", min_value=0, max_value=500, value=10)
    temp = st.number_input("Suhu (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    hum = st.number_input("Kelembaban (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)

    if st.button("Prediksi Besok"):
        # Ambil data historis
        df = pd.read_csv(SHEET_URL)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # Tambah data hari ini
        new_row = {
            "timestamp": pd.to_datetime(today),
            "sku": sku,
            "price": price,
            "sold": sold,
            "temperature_c": temp,
            "humidity": hum
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Urutkan & buat fitur tambahan
        df = df.sort_values(["sku","timestamp"])
        for lag in [1,2,3,7,14]:
            df[f"lag_{lag}"] = df.groupby("sku")["sold"].shift(lag)
        for win in [3,7,14]:
            df[f"rolling_mean_{win}"] = df.groupby("sku")["sold"].shift(1).rolling(window=win, min_periods=1).mean()
            df[f"rolling_std_{win}"] = df.groupby("sku")["sold"].shift(1).rolling(window=win, min_periods=1).std()

        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"]>=5).astype(int)
        df["total_demand_day"] = df.groupby("timestamp")["sold"].transform("sum")
        df["sku_share"] = df["sold"] / (df["total_demand_day"]+1e-5)
        df["sku_encoded"] = le.transform(df["sku"])

        # Ambil row terakhir utk prediksi besok
        latest_features = df.drop(columns=["timestamp","sku","sold"]).iloc[-1:]
        latest_features = latest_features.reindex(columns=feature_names, fill_value=0)

        pred = predict_sales(latest_features)
        st.success(f"🔮 Prediksi Penjualan Besok untuk {sku}: {pred[0]:.2f} unit")
