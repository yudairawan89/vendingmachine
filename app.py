# ======================================
# === app.py: Vending Machine GUI Final Extended ===
# ======================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# ======================================
# === STEP 1: Konfigurasi & Load Model ===
# ======================================

# Load base models
rf = joblib.load("RandomForest_model.pkl")
svm = joblib.load("SVM_model.pkl")
xgb = joblib.load("XGBoost_model.pkl")

# Load meta LSTM (tanpa compile)
lstm_meta = tf.keras.models.load_model("stacking_lstm_meta.h5", compile=False)

# Load scaler, encoder, dan feature names
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

# Link Google Sheets (gunakan CSV export)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1AvzsaiDZqQ_0tR7S3_OYCqwIeTIz3GQ27ptvT4GWk6A/export?format=csv"


# ======================================
# === STEP 2: Fungsi Prediksi ===
# ======================================
def predict_sales(input_df):
    # Pastikan kolom sama dengan feature_names
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 Monitoring", 
    "📝 Manual Input", 
    "📊 Prediksi Besok",
    "📆 Prediksi Minggu Depan",
    "📅 Prediksi Bulan Depan"
])


# -------------------------------
# Tab 1: Monitoring
# -------------------------------
with tab1:
    st.subheader("Data Monitoring dari Google Sheet")

    try:
        df = pd.read_csv(SHEET_URL)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H.%M.%S", errors="coerce")

        if df["timestamp"].isnull().all():
            st.warning("⚠️ Data ada, tapi kolom timestamp kosong/tidak valid.")
        else:
            latest = df.sort_values("timestamp").iloc[-1]

            col1, col2, col3 = st.columns(3)
            col1.metric("Suhu (°C)", f"{latest['temperature_c']}")
            col2.metric("Kelembaban (%)", f"{latest['humidity']}")
            col3.metric("SKU", latest['sku'])

            st.dataframe(df.tail(10))

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
            st.success(f"🔮 Prediksi Penjualan: {round(pred[0], 2)} unit")

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
        st.success(f"🔮 Prediksi Penjualan (Manual): {round(pred[0], 2)} unit")


# -------------------------------
# Tab 3: Prediksi Besok
# -------------------------------
with tab3:
    st.subheader("Prediksi Penjualan Besok untuk Setiap SKU")
    st.caption("Silakan masukkan data penjualan *hari ini* untuk setiap SKU. Sistem akan pakai ini sebagai lag_1 untuk prediksi besok.")

    skus = le.classes_
    input_today = {}
    for sku in skus:
        c1, c2 = st.columns(2)
        with c1:
            price = st.number_input(f"Harga {sku}", min_value=1000, max_value=20000, value=10000, step=500)
        with c2:
            sold = st.number_input(f"Penjualan Hari Ini {sku}", min_value=0, max_value=100, value=1, step=1)
        input_today[sku] = {"price": price, "sold": sold}

    if st.button("Prediksi Besok"):
        preds = []
        tomorrow = pd.Timestamp.today() + pd.Timedelta(days=1)
        for sku, vals in input_today.items():
            try:
                sku_encoded = le.transform([sku])[0]
            except:
                sku_encoded = 0

            features = pd.DataFrame([{
                "avg_price": vals["price"],
                "day_of_week": tomorrow.dayofweek,
                "is_weekend": 1 if tomorrow.dayofweek >= 5 else 0,
                "sku_encoded": sku_encoded,
                "lag_1": vals["sold"],
                "lag_2": vals["sold"],
                "lag_3": vals["sold"],
                "lag_7": vals["sold"],
                "lag_14": vals["sold"],
                "rolling_mean_3": vals["sold"],
                "rolling_mean_7": vals["sold"],
                "rolling_mean_14": vals["sold"],
                "rolling_std_3": 0,
                "rolling_std_7": 0,
                "rolling_std_14": 0,
                "total_demand_day": vals["sold"],
                "sku_share": 1.0
            }])

            pred = predict_sales(features)
            preds.append({"SKU": sku, "Prediksi Besok": round(float(pred[0]), 2)})

        st.table(pd.DataFrame(preds))


# -------------------------------
# Tab 4: Prediksi Minggu Depan
# -------------------------------
with tab4:
    st.subheader("Prediksi Penjualan Minggu Depan (7 Hari ke Depan)")

    skus = le.classes_
    input_today = {}
    for sku in skus:
        c1, c2 = st.columns(2)
        with c1:
            price = st.number_input(f"Harga {sku} (Minggu Depan)", min_value=1000, max_value=20000, value=10000, step=500)
        with c2:
            sold = st.number_input(f"Penjualan Hari Ini {sku} (Basis Prediksi Minggu)", min_value=0, max_value=100, value=1, step=1)
        input_today[sku] = {"price": price, "sold": sold}

    if st.button("Prediksi Minggu Depan"):
        all_preds = []
        today = pd.Timestamp.today()
        for day_offset in range(1, 8):  # 7 hari
            target_day = today + pd.Timedelta(days=day_offset)
            daily_preds = {"Tanggal": target_day.strftime("%Y-%m-%d")}
            for sku, vals in input_today.items():
                try:
                    sku_encoded = le.transform([sku])[0]
                except:
                    sku_encoded = 0

                features = pd.DataFrame([{
                    "avg_price": vals["price"],
                    "day_of_week": target_day.dayofweek,
                    "is_weekend": 1 if target_day.dayofweek >= 5 else 0,
                    "sku_encoded": sku_encoded,
                    "lag_1": vals["sold"],
                    "lag_2": vals["sold"],
                    "lag_3": vals["sold"],
                    "lag_7": vals["sold"],
                    "lag_14": vals["sold"],
                    "rolling_mean_3": vals["sold"],
                    "rolling_mean_7": vals["sold"],
                    "rolling_mean_14": vals["sold"],
                    "rolling_std_3": 0,
                    "rolling_std_7": 0,
                    "rolling_std_14": 0,
                    "total_demand_day": vals["sold"],
                    "sku_share": 1.0
                }])

                pred = predict_sales(features)
                daily_preds[sku] = round(float(pred[0]), 2)

            all_preds.append(daily_preds)

        st.dataframe(pd.DataFrame(all_preds))


# -------------------------------
# Tab 5: Prediksi Bulan Depan
# -------------------------------
with tab5:
    st.subheader("Prediksi Penjualan Bulan Depan (30 Hari ke Depan)")

    skus = le.classes_
    input_today = {}
    for sku in skus:
        c1, c2 = st.columns(2)
        with c1:
            price = st.number_input(f"Harga {sku} (Bulan Depan)", min_value=1000, max_value=20000, value=10000, step=500)
        with c2:
            sold = st.number_input(f"Penjualan Hari Ini {sku} (Basis Prediksi Bulan)", min_value=0, max_value=100, value=1, step=1)
        input_today[sku] = {"price": price, "sold": sold}

    if st.button("Prediksi Bulan Depan"):
        all_preds = []
        today = pd.Timestamp.today()
        for day_offset in range(1, 31):  # 30 hari
            target_day = today + pd.Timedelta(days=day_offset)
            daily_preds = {"Tanggal": target_day.strftime("%Y-%m-%d")}
            for sku, vals in input_today.items():
                try:
                    sku_encoded = le.transform([sku])[0]
                except:
                    sku_encoded = 0

                features = pd.DataFrame([{
                    "avg_price": vals["price"],
                    "day_of_week": target_day.dayofweek,
                    "is_weekend": 1 if target_day.dayofweek >= 5 else 0,
                    "sku_encoded": sku_encoded,
                    "lag_1": vals["sold"],
                    "lag_2": vals["sold"],
                    "lag_3": vals["sold"],
                    "lag_7": vals["sold"],
                    "lag_14": vals["sold"],
                    "rolling_mean_3": vals["sold"],
                    "rolling_mean_7": vals["sold"],
                    "rolling_mean_14": vals["sold"],
                    "rolling_std_3": 0,
                    "rolling_std_7": 0,
                    "rolling_std_14": 0,
                    "total_demand_day": vals["sold"],
                    "sku_share": 1.0
                }])

                pred = predict_sales(features)
                daily_preds[sku] = round(float(pred[0]), 2)

            all_preds.append(daily_preds)

        st.dataframe(pd.DataFrame(all_preds))
