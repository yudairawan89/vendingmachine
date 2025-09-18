# ===============================================
# === app.py: Vending Machine Monitoring & Prediction ===
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# ===============================================
# === STEP 1: Load Model & Preprocessor ===
# ===============================================

# Load base models
rf = joblib.load("RandomForest_model.pkl")
svm = joblib.load("SVM_model.pkl")
xgb = joblib.load("XGBoost_model.pkl")

# Load meta LSTM
lstm_meta = tf.keras.models.load_model("stacking_lstm_meta.h5", compile=False)

# Load scaler & encoder
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

# Link Google Sheets (CSV export)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1AvzsaiDZqQ_0tR7S3_OYCqwIeTIz3GQ27ptvT4GWk6A/export?format=csv"


# ===============================================
# === STEP 2: Fungsi Prediksi ===
# ===============================================
def predict_sales(input_df):
    # pastikan kolom sesuai training
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    X_scaled = scaler.transform(input_df)

    base_preds = [
        rf.predict(X_scaled).reshape(-1, 1),
        svm.predict(X_scaled).reshape(-1, 1),
        xgb.predict(X_scaled).reshape(-1, 1),
    ]
    meta_X = np.hstack(base_preds)
    meta_X_lstm = meta_X.reshape((meta_X.shape[0], 1, meta_X.shape[1]))

    y_pred = lstm_meta.predict(meta_X_lstm, verbose=0)
    return y_pred.flatten()


# ===============================================
# === STEP 3: Streamlit GUI ===
# ===============================================
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

        # Parsing timestamp (format pakai titik di jam)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="%Y-%m-%d %H.%M.%S", errors="coerce"
        )

        if df["timestamp"].isnull().all():
            st.warning("⚠️ Data ada, tapi kolom timestamp kosong/tidak valid.")
        else:
            latest = df.sort_values("timestamp").iloc[-1]

            col1, col2, col3 = st.columns(3)
            col1.metric("Suhu (°C)", f"{latest['temperature_c']}")
            col2.metric("Kelembaban (%)", f"{latest['humidity']}")
            col3.metric("SKU", latest["sku"])

            st.dataframe(df.tail(10))

            # Prediksi SKU terakhir
            try:
                sku_encoded = le.transform([latest["sku"]])[0]
            except:
                sku_encoded = 0

            features = pd.DataFrame(
                [
                    {
                        "avg_price": latest["price"],
                        "day_of_week": latest["timestamp"].dayofweek,
                        "is_weekend": 1
                        if latest["timestamp"].dayofweek >= 5
                        else 0,
                        "sku_encoded": sku_encoded,
                    }
                ]
            )

            pred = predict_sales(features)
            st.success(f"🔮 Prediksi Penjualan (SKU {latest['sku']}): {pred[0]:.2f} unit")

            # ======================================
            # Prediksi Harian Semua SKU
            # ======================================
            unique_skus = df["sku"].unique()

            preds_list = []
            for sku in unique_skus:
                try:
                    sku_encoded = le.transform([sku])[0]
                except:
                    sku_encoded = 0

                features = pd.DataFrame(
                    [
                        {
                            "avg_price": df[df["sku"] == sku]["price"].mean(),
                            "day_of_week": latest["timestamp"].dayofweek,
                            "is_weekend": 1
                            if latest["timestamp"].dayofweek >= 5
                            else 0,
                            "sku_encoded": sku_encoded,
                        }
                    ]
                )

                pred_val = predict_sales(features)[0]
                preds_list.append({"SKU": sku, "Prediksi Penjualan": round(pred_val, 2)})

            preds_df = pd.DataFrame(preds_list)

            st.subheader("🔮 Prediksi Penjualan Harian Semua SKU")
            st.dataframe(preds_df)

            st.bar_chart(preds_df.set_index("SKU"))

    except Exception as e:
        st.error(f"Gagal mengambil data Google Sheet: {e}")


# -------------------------------
# Tab 2: Manual Input
# -------------------------------
with tab2:
    st.subheader("Input Manual untuk Uji Coba Prediksi")

    sku = st.text_input("SKU (nama produk)", "Nescafe")
    avg_price = st.number_input(
        "Harga Rata-rata", min_value=1000, max_value=20000, value=10000, step=500
    )
    day_of_week = st.selectbox(
        "Hari ke-",
        list(range(7)),
        format_func=lambda x: ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][x],
    )
    is_weekend = 1 if day_of_week >= 5 else 0

    try:
        sku_encoded = le.transform([sku])[0]
    except:
        sku_encoded = 0

    if st.button("Prediksi Penjualan"):
        input_manual = pd.DataFrame(
            [
                {
                    "avg_price": avg_price,
                    "day_of_week": day_of_week,
                    "is_weekend": is_weekend,
                    "sku_encoded": sku_encoded,
                }
            ]
        )
        pred = predict_sales(input_manual)
        st.success(f"🔮 Prediksi Penjualan (Manual): {pred[0]:.2f} unit")
