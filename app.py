# ======================================
# === app.py: Vending Machine GUI Final ===
# ======================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# ======================================
# === STEP 1: Load Models & Config ===
# ======================================

# Load base models
rf = joblib.load("RandomForest_model.pkl")
svm = joblib.load("SVM_model.pkl")
xgb = joblib.load("XGBoost_model.pkl")

# Load meta LSTM
lstm_meta = tf.keras.models.load_model("stacking_lstm_meta.h5", compile=False)

# Load scaler, label encoder, dan feature_names
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

# Link Google Sheets (gunakan CSV export)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1AvzsaiDZqQ_0tR7S3_OYCqwIeTIz3GQ27ptvT4GWk6A/export?format=csv"


# ======================================
# === STEP 2: Fungsi Prediksi ===
# ======================================
def predict_sales(input_df):
    # Pastikan urutan kolom sesuai
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Isi NaN dengan 0 biar stabil
    input_df = input_df.fillna(0)

    # Scaling
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

tab1, tab2, tab3 = st.tabs(["📡 Monitoring", "📝 Manual Input", "🔮 Prediksi Besok"])

# -------------------------------
# Tab 1: Monitoring
# -------------------------------
with tab1:
    st.subheader("Data Monitoring dari Google Sheet")

    try:
        df = pd.read_csv(SHEET_URL)

        # Parsing timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        if df["timestamp"].isnull().all():
            st.warning("⚠️ Data ada, tapi kolom timestamp kosong/tidak valid.")
        else:
            latest = df.sort_values("timestamp").iloc[-1]

            # Tampilkan info dasar (tanpa suhu & kelembaban)
            col1, col2 = st.columns(2)
            col1.metric("SKU", latest['sku'])
            col2.metric("Harga", f"{latest['price']}")

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
            st.success(f"🔮 Prediksi Penjualan Hari Ini untuk {latest['sku']}: {pred[0]:.2f} unit")

            # Prediksi semua SKU
            st.markdown("### 🤯 Prediksi Penjualan Harian Semua SKU")
            all_preds = []
            for sku in le.classes_:
                sku_encoded = le.transform([sku])[0]
                feat = pd.DataFrame([{
                    "avg_price": latest["price"],
                    "day_of_week": latest["timestamp"].dayofweek,
                    "is_weekend": 1 if latest["timestamp"].dayofweek >= 5 else 0,
                    "sku_encoded": sku_encoded
                }])
                yhat = predict_sales(feat)
                all_preds.append({"SKU": sku, "Prediksi Penjualan": round(yhat[0], 2)})

            st.dataframe(pd.DataFrame(all_preds))

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
        st.success(f"🔮 Prediksi Penjualan (Manual) untuk {sku}: {pred[0]:.2f} unit")


# -------------------------------
# Tab 3: Prediksi Besok
# -------------------------------
with tab3:
    st.subheader("Prediksi Penjualan Besok untuk Setiap SKU")

    try:
        df = pd.read_csv(SHEET_URL)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        if df.empty or df["timestamp"].isnull().all():
            st.warning("⚠️ Tidak ada data valid untuk prediksi besok.")
        else:
            latest = df.sort_values("timestamp").iloc[-1]

            # Prediksi untuk semua SKU (besok)
            besok_preds = []
            besok_day = latest["timestamp"].dayofweek + 1
            if besok_day > 6:
                besok_day = 0
            is_weekend = 1 if besok_day >= 5 else 0

            for sku in le.classes_:
                sku_encoded = le.transform([sku])[0]
                feat = pd.DataFrame([{
                    "avg_price": latest["price"],
                    "day_of_week": besok_day,
                    "is_weekend": is_weekend,
                    "sku_encoded": sku_encoded
                }])
                yhat = predict_sales(feat)
                besok_preds.append({"SKU": sku, "Prediksi Besok": round(yhat[0], 2)})

            st.dataframe(pd.DataFrame(besok_preds))

    except Exception as e:
        st.error(f"Gagal menghitung prediksi besok: {e}")
