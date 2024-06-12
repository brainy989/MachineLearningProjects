import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
today = date.today()

d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=365)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

data = yf.download('GOOG', 
                   start=start_date, 
                   end=end_date, 
                   progress=False)
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)

data = data[["Date", "Close"]]

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# plt.figure(figsize=(15, 10))
# plt.plot(data["Date"], data["Close"])
# plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
# result = seasonal_decompose(data["Close"], model='multiplicative', period=30)
# fig = plt.figure()
# fig = result.plot()
# fig.set_size_inches(15, 10)

# pd.plotting.autocorrelation_plot(data["Close"])

# plot_pacf(data["Close"], lags=100)

# plt.show()

p, d, q = 5, 1, 1
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import warnings

model = sm.tsa.statespace.SARIMAX(data['Close'],
                                  order=(p, d, q),
                                  seasonal_order=(p, d, q, 12))
model = model.fit()

predictions = model.predict(len(data), len(data) + 10)
data["Close"].plot(legend=True, label="Training Data", figsize=(15, 10))
predictions.plot(legend=True, label="Predictions")

plt.show()