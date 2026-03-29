future_days = 20
future_predictions = []
temp_data = data.copy()
alpha = 0.7
for i in range(future_days):
    last_scaled = scaler.transform(temp_data)[-60:]
    pred = model.predict(last_scaled.reshape(1,60,3), verbose=0)
    pred_price = scaler.inverse_transform(
        np.concatenate((pred, np.zeros((1,2))), axis=1)
    )[0,0]
    if len(future_predictions) > 0:
        pred_price = alpha * pred_price + (1-alpha) * future_predictions[-1]
    future_predictions.append(pred_price)
    new_row = pd.DataFrame({'Close':[pred_price]})
    temp_data = pd.concat([temp_data, new_row], ignore_index=True)
    temp_data['SMA_short'] = temp_data['Close'].rolling(window=short_window).mean()
    temp_data['SMA_long'] = temp_data['Close'].rolling(window=long_window).mean()
    temp_data = temp_data.bfill()
future_prices = np.array(future_predictions)


plt.figure(figsize=(12,6))
plt.plot(data['Close'].values, label='Actual Close')
future_start = len(data['Close'])
plt.plot(range(future_start, future_start + future_days),
         future_prices, label='Future Prediction')
plt.xlabel("Day Number")
plt.ylabel("Price")
plt.title("Future Stock Price Prediction")
plt.legend()
plt.show()
