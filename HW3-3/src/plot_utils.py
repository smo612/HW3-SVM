import numpy as np
import streamlit as st
import plotly.graph_objs as go

def plot_3d_scatter_with_hyperplane(x1, x2, x3, Y, coef, intercept):
    # 繪製 3D 散點圖
    scatter = go.Scatter3d(
        x=x1, y=x2, z=x3,
        mode='markers',
        marker=dict(
            color=Y,
            colorscale='Portland',
            size=5,
            opacity=0.8
        )
    )

    # 繪製分離超平面，使用單一顏色並增加透明度
    xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 10),
                         np.linspace(min(x2), max(x2), 10))
    zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]

    surface = go.Surface(
        x=xx, y=yy, z=zz,
        colorscale=[[0, 'lightblue'], [1, 'lightblue']],
        opacity=0.5,
        showscale=False  # 移除色標
    )

    # 配置圖表
    layout = go.Layout(
        title='3D Scatter Plot with Y Color and Separating Hyperplane',
        scene=dict(
            xaxis_title='x1',
            yaxis_title='x2',
            zaxis_title='x3',
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig = go.Figure(data=[scatter, surface], layout=layout)
    
    # 在 Streamlit 中顯示 Plotly 圖
    st.plotly_chart(fig)