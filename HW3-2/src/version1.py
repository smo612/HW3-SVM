import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def plot_version1(num_points):
    # 設置隨機數種子
    np.random.seed(0)

    # 使用正態分布生成 x1 和 x2，中心為 (0, 0)，方差為 10
    x1 = np.random.normal(0, np.sqrt(10), num_points)
    x2 = np.random.normal(0, np.sqrt(10), num_points)

    # 計算每個點到原點的距離
    distances = np.sqrt(x1**2 + x2**2)

    # 根據距離標記類別，分界距離設為4
    Y = np.where(distances < 4, 0, 1)

    # 可視化數據
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x1[Y == 0], x2[Y == 0], color='blue', marker='o', label='Y=0')
    ax.scatter(x1[Y == 1], x2[Y == 1], color='red', marker='s', label='Y=1')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('2D Dataset with Distance-based Labels')
    ax.legend()
    ax.grid()
    ax.axis('equal')  # 確保圖形比例相等
    
    # 在 Streamlit 中顯示 Matplotlib 圖
    st.pyplot(fig)