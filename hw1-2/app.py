from flask import Flask, render_template, request
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

app = Flask(__name__)

# 讀取 CSV 資料並進行預測的核心邏輯
def forecast_stock_price():
    # 讀取 CSV 資料
    file_path = "C:\\Users\\py\\Desktop\\hw1-2\\2330-training.csv"
    data = pd.read_csv(file_path)

    # 資料處理
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
    data['y'] = data['y'].str.replace(',', '').astype(float)

    # 準備資料
    df = data.rename(columns={'Date': 'ds', 'y': 'y'})

    # 設定 Prophet 模型
    model = Prophet(changepoint_prior_scale=0.5, interval_width=0.95)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    # 訓練模型
    model.fit(df)

    # 預測未來 60 天
    future = model.make_future_dataframe(periods=60)
    forecast = model.predict(future)

    # 繪製圖表
    plt.figure(figsize=(14, 7))
    plt.plot(df['ds'], df['y'], color='black', label='Actual Data')
    plt.plot(forecast['ds'], forecast['yhat'], color='blue', label='Forecast')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.5)

    # 添加水平灰色虛線表示歷史平均
    historical_average = df['y'].mean()
    plt.axhline(y=historical_average, color='gray', linestyle='--', label='Historical Average')

    # 添加紅色虛線和箭頭標註預測初始化點
    plt.axvline(x=forecast['ds'][df.shape[0]], color='red', linestyle='--')
    plt.annotate('Forecast Initialization', xy=(forecast['ds'][df.shape[0]], forecast['yhat'][df.shape[0]]), 
                 xytext=(forecast['ds'][df.shape[0]], forecast['yhat'][df.shape[0]] + 5),
                 arrowprops=dict(facecolor='orange', shrink=0.05))

    # 添加綠色箭頭標註上升趨勢
    plt.annotate('Upward Trend', xy=(forecast['ds'][df.shape[0]//2], forecast['yhat'][df.shape[0]//2]), 
                 xytext=(forecast['ds'][df.shape[0]//2], forecast['yhat'][df.shape[0]//2] + 10),
                 arrowprops=dict(facecolor='green', shrink=0.05))

    # 設置圖表格式和標註
    plt.title('Stock Price Forecast with Prophet')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # 將圖像保存到內存中
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url

# 定義網站首頁路由
@app.route('/')
def index():
    plot_url = forecast_stock_price()  # 生成預測圖表
    return render_template('index.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)