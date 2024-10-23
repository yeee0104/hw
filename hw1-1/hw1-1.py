from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        a = float(request.form['slope'])
        noise_level = float(request.form['noise'])
        num_points = int(request.form['num_points'])

        # 生成數據
        x = np.linspace(0, 10, num_points)
        noise = np.random.normal(0, noise_level, num_points)
        y = a * x + noise

        # 擬合線性回歸模型
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        y_pred = model.predict(x.reshape(-1, 1))

        # 計算均方誤差
        mse = mean_squared_error(y, y_pred)

        # 繪製圖形
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='blue', label='Data Points')
        plt.plot(x, y_pred, color='red', label='Regression Line')
        plt.title('Linear Regression')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

        # 保存圖像到內存
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('index.html', mse=mse, plot_url=plot_url)

    return render_template('index.html', mse=None, plot_url=None)

if __name__ == '__main__':
    app.run(debug=True)
