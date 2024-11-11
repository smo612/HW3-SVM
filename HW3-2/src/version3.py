import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC

def plot_version3(num_points):
    # 設置隨機數種子
    np.random.seed(0)

    # 使用正態分布生成 x1 和 x2，中心為 (0, 0)，方差為 10
    x1 = np.random.normal(0, np.sqrt(10), num_points)
    x2 = np.random.normal(0, np.sqrt(10), num_points)

    # 計算每個點到原點的距離
    distances = np.sqrt(x1**2 + x2**2)

    # 根據距離標記類別
    labels = np.where(distances < 4, 0, 1)

    # 計算 x3 值，使用高斯函數
    x3 = np.exp(-0.1 * (x1**2 + x2**2))

    # 將 x1, x2, x3 堆疊成特徵矩陣 X
    X = np.column_stack((x1, x2, x3))

    # 使用 LinearSVC 訓練模型以找到分離超平面
    clf = LinearSVC(random_state=0, max_iter=10000)
    clf.fit(X, labels)
    coef = clf.coef_[0]
    intercept = clf.intercept_

    # 可視化數據與分離超平面
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 繪製數據點
    ax.scatter(x1[labels == 0], x2[labels == 0], x3[labels == 0], color='blue', marker='o', label='Y=0')
    ax.scatter(x1[labels == 1], x2[labels == 1], x3[labels == 1], color='red', marker='s', label='Y=1')

    # 繪製分離超平面
    xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 10),
                         np.linspace(min(x2), max(x2), 10))
    zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]
    ax.plot_surface(xx, yy, zz, color='gray', alpha=0.5)

    # 添加坐標軸標籤和標題
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title('3D Scatter Plot with Separating Hyperplane')
    ax.legend()
    
    # 在 Streamlit 中顯示 Matplotlib 圖
    st.pyplot(fig)