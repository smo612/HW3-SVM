import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.svm import LinearSVC

def plot_version5(num_points):
    # 設置隨機數種子
    np.random.seed(0)

    # 使用均勻分布生成數據點，形成方形分布
    x1 = np.random.uniform(-10, 10, num_points)
    x2 = np.random.uniform(-10, 10, num_points)

    # 計算每個點到原點的距離並分配標籤
    Y = np.where((np.abs(x1) < 4) & (np.abs(x2) < 4), 0, 1)

    # 添加偏移中心的另一組數據點
    x1_extra = np.random.uniform(5, 15, num_points)
    x2_extra = np.random.uniform(5, 15, num_points)
    Y_extra = np.where((np.abs(x1_extra - 10) < 4) & (np.abs(x2_extra - 10) < 4), 0, 1)

    # 合併兩組數據
    x1 = np.concatenate((x1, x1_extra))
    x2 = np.concatenate((x2, x2_extra))
    Y = np.concatenate((Y, Y_extra))

    # 計算第三維度 x3
    x3 = np.exp(-0.15 * (x1**2 + x2**2))

    # 使用 LinearSVC 訓練模型以找到分離超平面
    X = np.column_stack((x1, x2, x3))
    clf = LinearSVC(random_state=0, max_iter=10000, C=20)
    clf.fit(X, Y)
    coef = clf.coef_[0]
    intercept = clf.intercept_

    # 創建 Plotly 圖形
    fig = go.Figure()

    # 添加數據點
    fig.add_trace(go.Scatter3d(
        x=x1[Y == 0], y=x2[Y == 0], z=x3[Y == 0],
        mode='markers',
        marker=dict(color='blue', size=4),
        name='Y=0'
    ))

    fig.add_trace(go.Scatter3d(
        x=x1[Y == 1], y=x2[Y == 1], z=x3[Y == 1],
        mode='markers',
        marker=dict(color='red', size=4),
        name='Y=1'
    ))

    # 繪製分離超平面
    xx, yy = np.linspace(min(x1), max(x1), 10), np.linspace(min(x2), max(x2), 10)
    xx, yy = np.meshgrid(xx, yy)
    zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]

    fig.add_trace(go.Surface(x=xx, y=yy, z=zz, colorscale='Greys', opacity=0.5))

    # 設置圖形佈局
    fig.update_layout(
        scene=dict(
            xaxis_title='x1',
            yaxis_title='x2',
            zaxis_title='x3',
        ),
        title="3D Scatter Plot with Square Distribution and Separating Hyperplane (Plotly)"
    )

    # 在 Streamlit 中顯示 Plotly 圖
    st.plotly_chart(fig)