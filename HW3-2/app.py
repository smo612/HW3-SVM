import streamlit as st
from src.main import generate_data, train_model
from src.version1 import plot_version1
from src.version2 import plot_version2
from src.version3 import plot_version3
from src.version4 import plot_version4
from src.version5 import plot_version5

# streamlit run app.py

st.title("HW3-2. 2D SVM with streamlit deployment (3D plot) -dataset 分布在feature plane上圓形")

st.markdown("<br>", unsafe_allow_html=True)

# 生成數據和訓練模型
x1, x2, x3, Y = generate_data()
coef, intercept = train_model(x1, x2, x3, Y)

# 添加滑桿來調整數據點數量
num_points = st.slider("選擇數據點數量", min_value=100, max_value=1000, value=500, step=50)

st.markdown(f"""
使用 Python 生成一個 2D 數據集，數據點數量為 {num_points}。數據生成要求如下：

- **數據生成方式**：使用正態分布，中心為 (0, 0)，方差為 10，分別生成 x1 和 x2。
- **距離計算與標籤分配**：計算每個點到原點的距離，當距離小於 4 時，標記為 Y=0，否則標記為 Y=1。
- **可視化要求**：生成散點圖，用藍色表示 Y=0，紅色表示 Y=1，圖表應包括坐標軸標籤、圖例、標題以及相等的比例顯示。
""")
plot_version1(num_points) 


st.markdown("""
在現有的 x1 和 x2 基礎上生成 x3：使用高斯函數計算 x3，其中 x3 = exp(-0.1 * (x1**2 + x2**2))，這樣可以讓 x3 的值隨著距離增加而減小，形成錐形的分布效果。

3D 散點圖可視化：在 3D 空間中繪製 (x1, x2, x3)，以 Y 作為顏色標記，Y=0 類別使用藍色、Y=1 類別使用紅色。顯示坐標軸標籤 (x1, x2, x3)、圖例以及標題「3D Scatter Plot with Y Color」。
""")
plot_version2(num_points)


st.markdown("""
使用 x1, x2, x3 作為特徵，訓練 LinearSVC：將三個特徵堆疊成特徵矩陣 X，並使用 LinearSVC 訓練模型。獲取模型的分離超平面係數 coef 和截距 intercept。

可視化分離超平面：在3D散點圖中生成超平面，為此，先生成 x1 和 x2 的網格，然後基於支持向量機模型計算 x3 值（zz），並將結果顯示為灰色半透明平面。

圖形設置：確保 Y=0 類別使用藍色，Y=1 類別使用紅色，以便清晰展示分離超平面效果。
""")
plot_version3(num_points)


st.markdown("""
新增偏移中心的數據點：

使用中心為 (10, 10) 的正態分布生成額外的數據點，方差與原始數據一致。
將分界距離設為 4，距離小於 4 的點標記為 Y=0，其餘為 Y=1。
調整高斯函數：

將第三維 x3 的高斯函數衰減係數從 0.1 增大到 0.15，讓 x3 的範圍在 z 軸上更為顯著。
調整 SVM 模型的 C 值：

將 C 值設為 20，增強模型對錯誤分類的懲罰，使分離效果更加明顯。
""")
plot_version4(num_points)


fig = plot_version5(num_points)
st.plotly_chart(fig)