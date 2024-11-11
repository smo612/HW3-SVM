import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC

def plot_version3(num_points):
    np.random.seed(0)
    
    # 方形分布生成
    x1 = np.random.uniform(-5, 5, num_points)
    x2 = np.random.uniform(-5, 5, num_points)
    
    # 計算距離並分配標籤
    Y = np.where((np.abs(x1) < 2) & (np.abs(x2) < 2), 0, 1)
    
    # 計算第三維度 x3
    x3 = np.exp(-0.1 * (x1**2 + x2**2))
    
    # SVM 模型
    X = np.column_stack((x1, x2, x3))
    clf = LinearSVC(random_state=0, max_iter=10000)
    clf.fit(X, Y)
    coef = clf.coef_[0]
    intercept = clf.intercept_

    # 3D圖
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1[Y == 0], x2[Y == 0], x3[Y == 0], color='blue', label='Y=0')
    ax.scatter(x1[Y == 1], x2[Y == 1], x3[Y == 1], color='red', label='Y=1')
    
    # SVM 平面
    xx, yy = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
    zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]
    ax.plot_surface(xx, yy, zz, color='gray', alpha=0.5)
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title('3D Scatter Plot with Separating Hyperplane')
    ax.legend()
    
    plt.close(fig)  # 關閉圖像以避免 Streamlit 重複顯示
    return fig