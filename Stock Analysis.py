import os
import math as m
import statistics as s
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import io, sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

ML_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError:
    ML_AVAILABLE = False

# -------------------------------
# Helper function to build LSTM
# -------------------------------
def build_lstm_model(input_shape, units=50, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units // 2, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

# -------------------------------
# Main Analysis Function
# -------------------------------
def run_analysis(T, prd, p1, p2, seq_len, test_ratio, epochs, batch_size, future_days):
    logs = io.StringIO()
    sys.stdout = logs
    try:
        date_end = dt.date.today() - relativedelta(days=1)
        date_str_end = date_end.strftime("%d-%m-%y")
        date1y = dt.date.today() - relativedelta(years=1) + relativedelta(days=1)
        print(date1y.strftime("%d-%m-%y"), "to", date_str_end)

        D = yf.download(T, period=prd, progress=False)

        if "Close" in D.columns:
            close_series = D["Close"]
        elif ("Close", T) in D.columns:
            close_series = D[("Close", T)]

        CL = close_series.squeeze().tolist()
        xCL = [i + 1 for i in range(len(CL))]

        # Moving averages
        x1, x2, y1, y2 = [], [], [], []
        for i in range(p1, len(CL) + 1):
            y1.append(s.mean(CL[i - p1 : i]))
            x1.append(i)
        for i in range(p2, len(CL) + 1):
            y2.append(s.mean(CL[i - p2 : i]))
            x2.append(i)

        # ---------------- Plot Moving Averages ----------------
        plt.figure(figsize=(8, 4))
        plt.plot(xCL, CL, color="blue", label="Closing Price")
        if y1:
            plt.plot(x1, y1, color="r", label=f"MA - {p1} Days")
        if y2:
            plt.plot(x2, y2, color="g", label=f"MA - {p2} Days")

        plt.legend()
        plt.xlabel(f"Day count from {date1y}")
        plt.ylabel("Price")
        plt.title(f"{T} | MA({p1},{p2}) | upto {date_str_end}")
        plt.grid(True)
        tab1.pyplot(plt)

        # ---------------- LSTM ----------------
        if ML_AVAILABLE:
            prices = np.array(CL).reshape(-1, 1)
            scaler = MinMaxScaler((0, 1))
            prices_scaled = scaler.fit_transform(prices)

            def create_sequences(data, seq_length):
                X, y = [], []
                for i in range(seq_length, len(data)):
                    X.append(data[i - seq_length : i, 0])
                    y.append(data[i, 0])
                X, y = np.array(X), np.array(y)
                return X.reshape((X.shape[0], X.shape[1], 1)), y

            X, y = create_sequences(prices_scaled, seq_len)
            split = int(len(X) * (1 - test_ratio))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = build_lstm_model((seq_len, 1))
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

            pred_test = scaler.inverse_transform(model.predict(X_test))
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

            # ---------------- Plot Test Predictions ----------------
            plt.figure(figsize=(8, 4))
            plt.plot(range(len(CL)), CL, color="blue", label="Historical")
            start_idx = len(CL) - len(y_test_inv)
            plt.plot(range(start_idx, len(CL)), y_test_inv, color="orange", label="Actual present data")
            plt.plot(range(start_idx, len(CL)), pred_test, color="green", label="Predicted data")
            plt.legend()
            plt.title(f"{T} | LSTM Test Predictions")
            plt.xlabel(f"Day count from {date1y}")
            plt.ylabel("Price")
            plt.grid(True)
            tab2.pyplot(plt)

            # ---------------- FUTURE PREDICTION ----------------
            future_predictions = []
            temp_data = prices_scaled.tolist()

            for i in range(future_days):
                seq = np.array(temp_data[-seq_len:])
                seq = seq.reshape(1, seq_len, 1)
                pred = model.predict(seq, verbose=0)
                temp_data.append(pred[0].tolist())
                future_predictions.append(pred[0][0])

            future_predictions = scaler.inverse_transform(
                np.array(future_predictions).reshape(-1, 1)
            )

            # ---------------- Future Plot ----------------
            plt.figure(figsize=(8, 4))
            plt.plot(range(len(CL)), CL, label="Historical Price")
            future_x = range(len(CL), len(CL) + future_days)
            plt.plot(future_x, future_predictions, label="Future Prediction")
            plt.legend()
            plt.title(f"{T} Future Price Prediction")
            plt.xlabel("Day Number")
            plt.ylabel("Price")
            plt.grid(True)
            tab2.pyplot(plt)

            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test_inv, pred_test))
            r2 = r2_score(y_test_inv, pred_test)
            accuracy = max(0, r2) * 100

            tab3.write(f"📌 Accuracy: {accuracy:.2f}%")
            tab3.write(f"📌 RMSE: {rmse:.2f}")
            tab3.write(f"📌 R² Score: {r2:.3f}")

    except Exception as e:
        print("Error occurred:", e)

    sys.stdout = sys.__stdout__
    return logs.getvalue()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="📈 Stock Analyzer", page_icon="📊", layout="wide")

st.title("📈 Stock Analyzer")

st.sidebar.header("⚙️ Parameters")
labels = ["Ticker", "Period", "MA-1", "MA-2", "Seq Len", "Test Ratio", "Epochs", "Batch Size", "Future Days"]
defaults = ["TSLA", "3y", "50", "200", "30", "0.2", "25", "16", "10"]
entries = {}
for lbl, dft in zip(labels, defaults):
    entries[lbl] = st.sidebar.text_input(lbl, dft)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Moving Averages", "🤖 LSTM Predictions", "🔮 Future Prediction", "💰 Profit & Accuracy", "📜 Logs"]
)

if st.button("🚀 Run Analysis"):
    with st.spinner("⚙️ Running analysis... please wait"):
        logs = run_analysis(
            entries["Ticker"].upper().strip(),
            entries["Period"].strip(),
            int(entries["MA-1"]),
            int(entries["MA-2"]),
            int(entries["Seq Len"]),
            float(entries["Test Ratio"]),
            int(entries["Epochs"]),
            int(entries["Batch Size"]),
            int(entries["Future Days"])
        )
    tab4.text(logs)
