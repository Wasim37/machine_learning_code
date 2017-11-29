# encoding=utf-8

import pandas as pd
import numpy as np
import time
import sklearn
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 引入文件
path = 'notebook\datas\household_power_consumption_1000.txt'
names = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
         'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

# 读取数据
df = pd.read_csv(path, sep=';')

# 打印了前五行数据
print(df.head())

# 看所有的变量值
# for i in df.columns:
#     print(df[i])
# print(df[i].value_counts())

# 空值的处理
new_df = df.replace('?', np.nan)
datas = new_df.dropna(how='any')

print(df.head())


# 创建一个时间字符串格式化
def date_format(dt):
    import time
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


# 获取X，Y变量，讲时间转换为数值型的连续变量
# apply 对X的行数据应用函数
X = datas[names[0:2]]
X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)
Y = datas[names[2]]

print(X.head(5))
print(Y.head(5))

# 对数据集进行训练集、测试集划分
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导
ss = StandardScaler()
# fit_transform()先训练拟合数据，再标准化
X_train = ss.fit_transform(X_train)
# 标准化数据
X_test = ss.transform(X_test)

# 训练模型
lr = LinearRegression()
lr.fit(X_train, Y_train)

# 模型检验
print("准确率：", lr.score(X_train, Y_train))

# 预测y值
y_predict = lr.predict(X_test)

from sklearn.externals import joblib

# 模型保存, 持久化到硬盘
# http://blog.csdn.net/xierhacker/article/details/70495775
joblib.dump(ss, "data_ss.model")
joblib.dump(lr, "data_lr.model")
# 反序列化加载存储的模型
joblib.load("data_ss.model")
joblib.load("data_lr.model")

# 设置字符集，防止中文乱码
# rc参数配置详解：http://blog.csdn.net/xqf1528399071/article/details/53385593
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 预测值与实际值画(图比较
t = np.arange(len(X_test))
plt.figure(facecolor='w')
plt.plot(t, Y_test, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, y_predict, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc='lower right')
plt.title(u'线性回归预测时间与功率之间的关系', fontsize=20)
plt.grid(b=True)
plt.show()
