# ======================================
# === app.py: Vending Machine GUI Final All-in-One (Monitoring Fresh) ===
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
SHEET_VIEW_URL = "https://docs.google.com/spreadsheets/d/1AvzsaiDZqQ_0tR7S3_OYCqwIeTIz3GQ27ptvT4GWk6A/edit"


# ======================================
# === STEP 2: Fungsi Prediksi ===
# ======================================
def predict_sales(input_df):
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📡 Monitoring",
    "🔮 Prediksi per Produk",
    "📊 Prediksi Besok",
    "📅 Prediksi Minggu Depan",
    "📆 Prediksi Bulan Depan",
    "📂 Prediksi dari File XLSX"
])


# -------------------------------
# Tab 1: Monitoring (Fresh)
# -------------------------------
with tab1:
    st.subheader("📡 Monitoring Harian")

    try:
        df = pd.read_csv(SHEET_URL)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H.%M.%S", errors="coerce")

        if df["timestamp"].isnull().all():
            st.warning("⚠️ Data ada, tapi timestamp tidak valid.")
        else:
            today = pd.Timestamp.today().date()
            df_today = df[df["timestamp"].dt.date == today]

            if df_today.empty:
                st.warning("⚠️ Tidak ada data penjualan untuk hari ini.")
            else:
                # Info suhu & kelembaban terakhir
                latest = df_today.sort_values("timestamp").iloc[-1]
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("🌡️ Suhu (°C)", f"{latest['temperature_c']}")
                with c2:
                    st.metric("💧 Kelembaban (%)", f"{latest['humidity']}")

                # Agregasi penjualan per SKU
                sales_today = df_today.groupby("sku")["sold"].sum().reset_index()
                preds = []
                tomorrow = pd.Timestamp.today() + pd.Timedelta(days=1)

                for _, row in sales_today.iterrows():
                    try:
                        sku_encoded = le.transform([row["sku"]])[0]
                    except:
                        sku_encoded = 0

                    features = pd.DataFrame([{
                        "avg_price": df_today[df_today["sku"] == row["sku"]]["price"].mean(),
                        "day_of_week": tomorrow.dayofweek,
                        "is_weekend": 1 if tomorrow.dayofweek >= 5 else 0,
                        "sku_encoded": sku_encoded,
                        "lag_1": row["sold"]
                    }])

                    pred = predict_sales(features)
                    preds.append({
                        "SKU": row["sku"],
                        "Terjual Hari Ini": int(row["sold"]),
                        "Prediksi Besok": round(float(pred[0]), 2)
                    })

                st.markdown("### 📊 Ringkasan Penjualan Harian & Prediksi Besok")
                st.dataframe(pd.DataFrame(preds), use_container_width=True)

                st.markdown(
                    f"<a href='{SHEET_VIEW_URL}' target='_blank'>"
                    "<button style='background-color:#4CAF50; color:white; padding:10px 20px; border:none; border-radius:5px;'>"
                    "📂 Database Penjualan</button></a>",
                    unsafe_allow_html=True
                )

    except Exception as e:
        st.error(f"Gagal mengambil data Google Sheet: {e}")


# -------------------------------
# Tab 2: Prediksi per Produk
# -------------------------------
with tab2:
    st.subheader("Prediksi per Produk (Harian, Mingguan, Bulanan)")

    sku = st.selectbox("Pilih Produk (SKU)", le.classes_)
    avg_price = st.number_input("Harga Rata-rata", min_value=1000, max_value=20000, value=10000, step=500)

    pred_scope = st.radio("Pilih Periode Prediksi", ["Harian", "Mingguan", "Bulanan"])

    day_of_week = pd.Timestamp.today().dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0

    try:
        sku_encoded = le.transform([sku])[0]
    except:
        sku_encoded = 0

    if st.button("Prediksi Penjualan (per Produk)"):
        features = pd.DataFrame([{
            "avg_price": avg_price,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "sku_encoded": sku_encoded
        }])
        pred = predict_sales(features)

        factor = 1
        if pred_scope == "Mingguan":
            factor = 7
        elif pred_scope == "Bulanan":
            factor = 30

        st.success(f"🔮 Prediksi {pred_scope} untuk {sku}: {pred[0] * factor:.2f} unit")


# -------------------------------
# Tab 3: Prediksi Besok (semua SKU)
# -------------------------------
with tab3:
    st.subheader("Prediksi Penjualan Besok untuk Setiap SKU")

    skus = le.classes_
    input_today = {}
    for sku in skus:
        c1, c2 = st.columns(2)
        with c1:
            price = st.number_input(f"Harga {sku}", min_value=1000, max_value=20000, value=10000, step=500, key=f"price_{sku}_besok")
        with c2:
            sold = st.number_input(f"Penjualan Hari Ini {sku}", min_value=0, max_value=100, value=1, step=1, key=f"sold_{sku}_besok")
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
                "lag_1": vals["sold"]
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
            price = st.number_input(f"Harga {sku}", min_value=1000, max_value=20000, value=10000, step=500, key=f"price_{sku}_minggu")
        with c2:
            sold = st.number_input(f"Penjualan Hari Ini {sku}", min_value=0, max_value=100, value=1, step=1, key=f"sold_{sku}_minggu")
        input_today[sku] = {"price": price, "sold": sold}

    if st.button("Prediksi Minggu Depan"):
        preds = []
        for sku, vals in input_today.items():
            try:
                sku_encoded = le.transform([sku])[0]
            except:
                sku_encoded = 0

            features = pd.DataFrame([{
                "avg_price": vals["price"],
                "day_of_week": pd.Timestamp.today().dayofweek,
                "is_weekend": 1 if pd.Timestamp.today().dayofweek >= 5 else 0,
                "sku_encoded": sku_encoded,
                "lag_1": vals["sold"]
            }])

            pred = predict_sales(features)
            preds.append({"SKU": sku, "Prediksi Mingguan": round(float(pred[0] * 7), 2)})

        st.table(pd.DataFrame(preds))


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
            price = st.number_input(f"Harga {sku}", min_value=1000, max_value=20000, value=10000, step=500, key=f"price_{sku}_bulan")
        with c2:
            sold = st.number_input(f"Penjualan Hari Ini {sku}", min_value=0, max_value=100, value=1, step=1, key=f"sold_{sku}_bulan")
        input_today[sku] = {"price": price, "sold": sold}

    if st.button("Prediksi Bulan Depan"):
        preds = []
        for sku, vals in input_today.items():
            try:
                sku_encoded = le.transform([sku])[0]
            except:
                sku_encoded = 0

            features = pd.DataFrame([{
                "avg_price": vals["price"],
                "day_of_week": pd.Timestamp.today().dayofweek,
                "is_weekend": 1 if pd.Timestamp.today().dayofweek >= 5 else 0,
                "sku_encoded": sku_encoded,
                "lag_1": vals["sold"]
            }])

            pred = predict_sales(features)
            preds.append({"SKU": sku, "Prediksi Bulanan": round(float(pred[0] * 30), 2)})

        st.table(pd.DataFrame(preds))


# -------------------------------
# Tab 6: Prediksi dari File XLSX
# -------------------------------
with tab6:
    st.subheader("📂 Upload File XLSX untuk Prediksi")

    uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            st.success("✅ File berhasil dibaca")
            st.dataframe(df.head())

            mode = st.radio("Pilih Mode Prediksi", ["Harian", "Mingguan", "Bulanan"])

            preds = []
            tomorrow = pd.Timestamp.today() + pd.Timedelta(days=1)

            for _, row in df.iterrows():
                try:
                    sku_encoded = le.transform([row["sku"]])[0]
                except:
                    sku_encoded = 0

                features = pd.DataFrame([{
                    "avg_price": row["price"],
                    "day_of_week": tomorrow.dayofweek,
                    "is_weekend": 1 if tomorrow.dayofweek >= 5 else 0,
                    "sku_encoded": sku_encoded,
                    "lag_1": row["sold"]
                }])

                pred = predict_sales(features)
                factor = 1
                if mode == "Mingguan":
                    factor = 7
                elif mode == "Bulanan":
                    factor = 30

                preds.append({"SKU": row["sku"], f"Prediksi {mode}": round(float(pred[0] * factor), 2)})

            st.table(pd.DataFrame(preds))

        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
