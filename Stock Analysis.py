import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import RMSprop

st.set_page_config(layout="wide")

# ================= Sidebar =================
st.sidebar.title("Stock LSTM Analysis")

ticker = st.sidebar.text_input("Ticker", value="TSLA")
period = st.sidebar.text_input("Period", value="3y")

ma_type = st.sidebar.selectbox("MA Type", ["SMA", "EMA"])
ma1 = st.sidebar.number_input("MA 1", value=40)
ma2 = st.sidebar.number_input("MA 2", value=120)

seq_len = st.sidebar.number_input("Sequence Length", value=60)
units = st.sidebar.number_input("LSTM Units", value=60)
epochs = st.sidebar.number_input("Epochs", value=50)
batch_size = st.sidebar.number_input("Batch Size", value=32)
test_ratio = st.sidebar.number_input("Test Ratio", value=0.2)
future_days = st.sidebar.number_input("Future Prediction Days", value=30)

run = st.sidebar.button("Run Analysis")

# ================= Tabs =================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Closing + MA + RSI",
    "LSTM Training",
    "Future Prediction",
    "Metrics",
    "Logs"
])

# ================= Functions =================
def compute_ma(series, period, mode):
    if mode == "SMA":
        return series.rolling(window=period).mean()
    else:
        return series.ewm(span=period, adjust=False).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period).mean()
    avg_loss = loss.ewm(alpha=1/period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ================= Run Analysis =================
if run:

    logs = []

    # ===== Download Data =====
    logs.append("Downloading data...")
    data = yf.download(ticker, period=period)
    data = data[['Close']].dropna()

    # ===== Indicators =====
    logs.append("Calculating indicators...")
    data['MA1'] = compute_ma(data['Close'], ma1, ma_type)
    data['MA2'] = compute_ma(data['Close'], ma2, ma_type)
    data['RSI'] = compute_rsi(data['Close'])

    data.dropna(inplace=True)

    # ===== Tab 1 Plot =====
    with tab1:
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8), sharex=True)

        ax1.plot(data['Close'], label='Close')
        ax1.plot(data['MA1'], label='MA1')
        ax1.plot(data['MA2'], label='MA2')

        # Crossovers
        diff = data['MA1'] - data['MA2']
        cross = np.where(np.diff(np.sign(diff)))[0]

        for c in cross:
            if diff.iloc[c] > 0:
                ax1.scatter(data.index[c], data['MA1'].iloc[c], color='green', marker='^')
            else:
                ax1.scatter(data.index[c], data['MA1'].iloc[c], color='red', marker='v')

        ax1.legend()
        ax2.plot(data['RSI'], color='purple')
        ax2.axhline(70, linestyle='--')
        ax2.axhline(30, linestyle='--')

        st.pyplot(fig)

    # ===== LSTM Dataset =====
    logs.append("Preparing dataset...")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[['Close','MA1','MA2','RSI']])

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i,0])

    X, y = np.array(X), np.array(y)

    split = int(len(X)*(1-test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ===== LSTM Model =====
    logs.append("Training LSTM...")
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(seq_len,4)))
    model.add(LSTM(units))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer=RMSprop(), loss='mse')

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=0
    )

    # ===== Predictions =====
    logs.append("Predicting test set...")
    preds = model.predict(X_test)

    # Inverse scale
    pred_prices = []
    for i in range(len(preds)):
        temp = np.zeros((1,4))
        temp[0,0] = preds[i]
        pred_prices.append(scaler.inverse_transform(temp)[0][0])

    actual_prices = data['Close'].values[-len(pred_prices):]

    # ===== Tab 2 Plot =====
    with tab2:
        fig = plt.figure(figsize=(12,6))
        plt.plot(data['Close'], label='Close')
        plt.plot(data.index[-len(pred_prices):], pred_prices, label='Predicted')
        plt.axvline(data.index[split], linestyle='--')
        plt.legend()
        st.pyplot(fig)

    # ===== Future Prediction =====
    logs.append("Predicting future prices...")
    future_preds = []
    temp_data = scaled.copy()

    for i in range(future_days):
        last_seq = temp_data[-seq_len:]
        pred = model.predict(last_seq.reshape(1,seq_len,4))
        future_preds.append(pred[0][0])

        new_row = np.array([[pred[0][0],0,0,50]])
        temp_data = np.vstack((temp_data,new_row))

    future_prices = []
    for p in future_preds:
        temp = np.zeros((1,4))
        temp[0,0] = p
        future_prices.append(scaler.inverse_transform(temp)[0][0])

    # ===== Tab 3 Plot =====
    with tab3:
        fig = plt.figure(figsize=(12,6))
        plt.plot(data['Close'], label='Past')
        future_index = pd.date_range(start=data.index[-1], periods=future_days+1)[1:]
        plt.plot(future_index, future_prices, label='Future')
        plt.axvline(data.index[-1], linestyle='--')
        plt.legend()
        st.pyplot(fig)

    # ===== Metrics =====
    rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    r2 = r2_score(actual_prices, pred_prices)
    mape = np.mean(np.abs((actual_prices - pred_prices)/actual_prices))*100
    accuracy = 100 - mape

    with tab4:
        st.metric("RMSE", round(rmse,2))
        st.metric("R2 Score", round(r2,3))
        st.metric("MAPE", round(mape,2))
        st.metric("Accuracy (%)", round(accuracy,2))

    # ===== Logs =====
    logs.append("Done.")

    with tab5:
        for l in logs:
            st.write(l)
