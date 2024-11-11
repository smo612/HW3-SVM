import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D

def plot_version2(num_points):
    # 設置隨機數種子
    np.random.seed(0)

    # 使用均勻分布生成 x1 和 x2，生成方形分布
    x1 = np.random.uniform(-5, 5, num_points)
    x2 = np.random.uniform(-5, 5, num_points)

    # 計算每個點到原點的距離
    distances = np.sqrt(x1**2 + x2**2)

    # 根據距離標記類別
    labels = np.where(distances < 4, 0, 1)

    # 計算 x3 值，使用高斯函數
    x3 = np.exp(-0.1 * (x1**2 + x2**2))

    # 可視化數據（3D 散點圖）
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 使用不同顏色標記類別
    ax.scatter(x1[labels == 0], x2[labels == 0], x3[labels == 0], color='blue', marker='o', label='Y=0')
    ax.scatter(x1[labels == 1], x2[labels == 1], x3[labels == 1], color='red', marker='s', label='Y=1')

    # 添加坐標軸標籤和標題
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title('3D Scatter Plot with Y Color')
    ax.legend()

    # 在 Streamlit 中顯示 Matplotlib 圖
    st.pyplot(fig)
