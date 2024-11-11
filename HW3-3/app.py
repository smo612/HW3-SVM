import streamlit as st
from src.main import generate_data, train_model
from src.version1 import plot_version1
from src.version2 import plot_version2
from src.version3 import plot_version3
from src.version4 import plot_version4
from src.version5 import plot_version5

# streamlit run app.py

st.title("HW3-3. 2D dataset 分布在feature plane上非圓形")

# 調整數據點數量的滑桿
num_points = st.slider("選擇數據點數量", min_value=100, max_value=1000, value=500, step=100)

# 生成數據和訓練模型
x1, x2, x3, Y = generate_data(num_points)
coef, intercept = train_model(x1, x2, x3, Y)

plot_version1(num_points)

plot_version2(num_points)

fig = plot_version3(num_points)
st.pyplot(fig)

plot_version4(num_points)

plot_version5(num_points)