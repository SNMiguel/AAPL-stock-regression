import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Download stock data for Apple - AAPL
data = yf.download("AAPL", start="2023-01-01", end="2024-01-01")

# 2. Use 'Close' price as target, and create a simple feature (time index)
data['Day'] = np.arange(len(data))  # simple time index

X = data[['Day']]  # features
y = data['Close']  # target

# 3. Split train-test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate with MAE and MSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# 7. Show prediction vs actual
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(data.index[-len(y_test):], y_test, label="Actual Price", color="blue")
plt.plot(data.index[-len(y_test):], y_pred, label="Predicted Price", color="red", linestyle="--")
plt.title("Stock Price Prediction with Linear Regression")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()