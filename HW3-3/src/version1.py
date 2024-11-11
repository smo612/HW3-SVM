import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def plot_version1(num_points):
    # 生成方形分布數據
    x1 = np.random.uniform(-5, 5, num_points)  # x1 坐標範圍 [-5, 5]
    x2 = np.random.uniform(-5, 5, num_points)  # x2 坐標範圍 [-5, 5]

    # 新的標籤分配：根據象限或區域劃分
    Y = np.where((x1 < 2) & (x1 > -2) & (x2 < 2) & (x2 > -2), 0, 1)  # 中心方形區域設為 Y=0，其餘為 Y=1

    # 可視化數據
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x1[Y == 0], x2[Y == 0], color='blue', marker='o', label='Y=0')
    ax.scatter(x1[Y == 1], x2[Y == 1], color='red', marker='s', label='Y=1')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('2D Dataset with Square Distribution')
    ax.legend()
    ax.grid()
    ax.axis('equal')  # 保持比例相等
    
    # 在 Streamlit 中顯示 Matplotlib 圖
    st.pyplot(fig)
