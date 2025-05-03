# -*- coding: utf-8 -*-

import pickle
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 创建一个简单的分类数据集
X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建一个简单的逻辑回归模型
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 将模型保存为.pkl文件
model_filename = 'simple_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)

model_filename

