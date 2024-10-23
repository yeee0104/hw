import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime
import matplotlib.dates as mdates

# 1. 讀取 CSV 資料
file_path = r"C:\Users\py\Desktop\hw1-2\2330-training.csv"
df = pd.read_csv(file_path)

# 2. 資料處理
# 將 'Date' 欄位轉換為日期格式
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# 移除 'y' 欄位中的逗號，並將其轉換為 float
df['y'] = df['y'].str.replace(',', '').astype(float)

# 3. 使用 Prophet 模型預測
# 準備 Prophet 模型所需的資料格式
df_prophet = df[['Date', 'y']].rename(columns={'Date': 'ds', 'y': 'y'})

# 建立 Prophet 模型，設定變化點靈敏度和不確定性區間
model = Prophet(changepoint_prior_scale=0.5, interval_width=0.95)
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# 訓練模型
model.fit(df_prophet)

# 4. 預測未來 60 天的股票價格
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

# 5. 繪製圖表
plt.figure(figsize=(10, 6))

# 繪製實際數據
plt.plot(df_prophet['ds'], df_prophet['y'], 'k-', label='Actual Data')

# 繪製預測數據和不確定性區間
plt.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecast')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='blue', alpha=0.2)

# 繪製歷史平均線
historical_avg = df_prophet['y'].mean()
plt.axhline(y=historical_avg, color='gray', linestyle='--', label='Historical Average')

# 在預測初始化點加紅色虛線
forecast_init_date = df_prophet['ds'].max()
plt.axvline(x=forecast_init_date, color='red', linestyle='--', label='Forecast Initialization')

# 添加綠色箭頭標註上升趨勢
plt.annotate('Upward Trend', xy=(forecast['ds'].iloc[-30], forecast['yhat'].iloc[-30]), 
             xytext=(forecast['ds'].iloc[-30], forecast['yhat'].iloc[-30] - 5),
             arrowprops=dict(facecolor='green', shrink=0.05, width=2), fontsize=12)

# 添加紅色箭頭標註預測初始化
plt.annotate('Forecast Initialization', xy=(forecast_init_date, forecast['yhat'].iloc[-60]),
             xytext=(forecast_init_date - pd.DateOffset(days=20), forecast['yhat'].iloc[-60] + 10),
             arrowprops=dict(facecolor='orange', shrink=0.05, width=2), fontsize=12)

# 設定日期格式
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# 添加標題和標籤
plt.title('Stock Price Forecast with Prophet')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# 顯示圖表
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
