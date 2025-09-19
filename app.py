# ======================================
# === app.py: Vending Machine GUI Final ===
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
    "🛒 Prediksi per Produk", 
    "📊 Prediksi Besok", 
    "📅 Prediksi Minggu Depan", 
    "📆 Prediksi Bulan Depan"
])


# -------------------------------
# Tab 1: Monitoring
# -------------------------------
with tab1:
    st.subheader("Data Monitoring dari Google Sheet")

    try:
        df = pd.read_csv(SHEET_URL)

        # Parsing timestamp manual (format pakai titik di jam)
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

            # Mapping SKU pakai LabelEncoder
            try:
                sku_encoded = le.transform([latest["sku"]])[0]
            except:
                sku_encoded = 0  # default kalau SKU tidak dikenal

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
# Tab 2: Prediksi Per Produk
# -------------------------------
with tab2:
    st.subheader("Prediksi Penjualan per Produk")

    # Dropdown SKU
    sku = st.selectbox("Pilih Produk (SKU)", le.classes_)
    avg_price = st.number_input("Harga Rata-rata", min_value=1000, max_value=20000, value=10000, step=500)

    # Pilih jenis prediksi
    pred_type = st.radio("Jenis Prediksi", ["Harian", "Mingguan", "Bulanan"])

    if pred_type == "Harian":
        day_of_week = st.selectbox(
            "Hari ke-",
            list(range(7)),
            format_func=lambda x: ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"][x]
        )
        is_weekend = 1 if day_of_week >= 5 else 0
    else:
        day_of_week = pd.Timestamp.today().dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0

    try:
        sku_encoded = le.transform([sku])[0]
    except:
        sku_encoded = 0

    if st.button("Prediksi Penjualan"):
        if pred_type == "Harian":
            input_manual = pd.DataFrame([{
                "avg_price": avg_price,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "sku_encoded": sku_encoded
            }])
            pred = predict_sales(input_manual)
            st.success(f"🔮 Prediksi Harian {sku}: {pred[0]:.2f} unit")

        elif pred_type == "Mingguan":
            preds = []
            for i in range(7):
                future_day = (pd.Timestamp.today() + pd.Timedelta(days=i+1)).dayofweek
                is_weekend = 1 if future_day >= 5 else 0
                features = pd.DataFrame([{
                    "avg_price": avg_price,
                    "day_of_week": future_day,
                    "is_weekend": is_weekend,
                    "sku_encoded": sku_encoded
                }])
                pred = predict_sales(features)
                preds.append(pred[0])
            st.info(f"📅 Prediksi Total Mingguan {sku}: {sum(preds):.2f} unit")

        elif pred_type == "Bulanan":
            preds = []
            for i in range(30):
                future_day = (pd.Timestamp.today() + pd.Timedelta(days=i+1)).dayofweek
                is_weekend = 1 if future_day >= 5 else 0
                features = pd.DataFrame([{
                    "avg_price": avg_price,
                    "day_of_week": future_day,
                    "is_weekend": is_weekend,
                    "sku_encoded": sku_encoded
                }])
                pred = predict_sales(features)
                preds.append(pred[0])
            st.info(f"📆 Prediksi Total Bulanan {sku}: {sum(preds):.2f} unit")


# -------------------------------
# Tab 3: Prediksi Besok per SKU
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

        result_df = pd.DataFrame(preds)
        st.table(result_df)


# -------------------------------
# Tab 4: Prediksi Minggu Depan
# -------------------------------
with tab4:
    st.subheader("Prediksi Minggu Depan untuk Setiap SKU")

    skus = le.classes_
    preds = []
    for sku in skus:
        try:
            sku_encoded = le.transform([sku])[0]
        except:
            sku_encoded = 0
        total_week = 0
        for i in range(7):
            future_day = (pd.Timestamp.today() + pd.Timedelta(days=i+1)).dayofweek
            is_weekend = 1 if future_day >= 5 else 0
            features = pd.DataFrame([{
                "avg_price": 10000,
                "day_of_week": future_day,
                "is_weekend": is_weekend,
                "sku_encoded": sku_encoded
            }])
            pred = predict_sales(features)
            total_week += pred[0]
        preds.append({"SKU": sku, "Prediksi Mingguan": round(float(total_week), 2)})

    result_df = pd.DataFrame(preds)
    st.table(result_df)


# -------------------------------
# Tab 5: Prediksi Bulan Depan
# -------------------------------
with tab5:
    st.subheader("Prediksi Bulan Depan untuk Setiap SKU")

    skus = le.classes_
    preds = []
    for sku in skus:
        try:
            sku_encoded = le.transform([sku])[0]
        except:
            sku_encoded = 0
        total_month = 0
        for i in range(30):
            future_day = (pd.Timestamp.today() + pd.Timedelta(days=i+1)).dayofweek
            is_weekend = 1 if future_day >= 5 else 0
            features = pd.DataFrame([{
                "avg_price": 10000,
                "day_of_week": future_day,
                "is_weekend": is_weekend,
                "sku_encoded": sku_encoded
            }])
            pred = predict_sales(features)
            total_month += pred[0]
        preds.append({"SKU": sku, "Prediksi Bulanan": round(float(total_month), 2)})

    result_df = pd.DataFrame(preds)
    st.table(result_df)
