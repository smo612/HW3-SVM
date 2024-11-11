import numpy as np
from sklearn.svm import LinearSVC

def generate_data(num_points=500):
    # 使用正態分布生成 x1 和 x2
    mean = 0
    variance = 10
    x1 = np.random.normal(mean, np.sqrt(variance), num_points)
    x2 = np.random.normal(mean, np.sqrt(variance), num_points)

    # 計算距離並分配標籤，將分界距離設為4
    distances = np.sqrt(x1**2 + x2**2)
    Y = np.where(distances < 4, 0, 1)

    # 添加偏移中心的另一組數據點
    x1_extra = np.random.normal(10, np.sqrt(variance), num_points)
    x2_extra = np.random.normal(10, np.sqrt(variance), num_points)
    distances_extra = np.sqrt((x1_extra - 10)**2 + (x2_extra - 10)**2)
    Y_extra = np.where(distances_extra < 4, 0, 1)

    # 合併兩組數據
    x1 = np.concatenate((x1, x1_extra))
    x2 = np.concatenate((x2, x2_extra))
    Y = np.concatenate((Y, Y_extra))

    # 計算第三維度 x3
    x3 = np.exp(-0.1 * (x1**2 + x2**2))
    
    return x1, x2, x3, Y

def train_model(x1, x2, x3, Y):
    X = np.column_stack((x1, x2, x3))
    clf = LinearSVC(random_state=0, max_iter=10000)
    clf.fit(X, Y)
    coef = clf.coef_[0]
    intercept = clf.intercept_[0]
    return coef, intercept
