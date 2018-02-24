# encoding=utf-8

# 引入所需要的全部包
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import time


# 创建一个时间字符串格式化字符串
def date_format(dt):
    import time
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 加载数据
# path = 'notebook\datas\household_power_consumption.txt' ## 全部数据
# path = 'notebook\datas\household_power_consumption_200.txt' ## 200行数据
path = 'notebook\datas\household_power_consumption_1000.txt'  ## 1000行数据
df = pd.read_csv(path, sep=';', low_memory=False)

# 日期、时间、有功功率、无功功率、电压、电流、厨房用电功率、洗衣服用电功率、热水器用电功率
names = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
         'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

# 异常数据处理(异常数据过滤)
new_df = df.replace('?', np.nan)
datas = new_df.dropna(how='any')  # 只要有列为空，就进行删除操作

# 时间和电压之间的关系(Linear)
# 获取x和y变量, 并将时间转换为数值型连续变量
X = datas[names[0:2]]
X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)
Y = datas[names[4]]

# print(X.head())
# print(Y.head())

# 对数据集进行测试集合训练集划分
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 数据标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)  # 训练并转换
X_test = ss.transform(X_test)  # 数据标准化操作

# 模型训练
lr = LinearRegression()
lr.fit(X_train, Y_train)  # 训练模型

# 模型校验
y_predict = lr.predict(X_test)  # 预测结果

# 模型效果
print("准确率:", lr.score(X_test, Y_test))

# 预测值和实际值画图比较(时间和电压的关系)
t = np.arange(len(X_test))
plt.figure(facecolor='w')
plt.plot(t, Y_test, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, y_predict, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc='lower right')
plt.title(u'线性回归预测时间和功率之间的关系', fontsize=20)
plt.grid(b=True)
plt.show()

# 2 时间和电压之间的关系(Linear-多项式)
models = [
    Pipeline([
        ('Poly', PolynomialFeatures()),
        ('Linear', LinearRegression())
    ])
]
model = models[0]
print(model)

# 获取x和y变量, 并将时间转换为数值型连续变量
X = datas[names[0:2]]
X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)
Y = datas[names[4]]

print(X.head())
print(Y.head())

# 对数据集进行测试集合训练集划分
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 数据标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)  # 训练并转换
print(X_train)
X_test = ss.transform(X_test)  # 直接使用在模型构建数据上进行一个数据标准化操作

# 模型训练
t = np.arange(len(X_test))
N = 5
d_pool = np.arange(1, N, 1)  # 阶 1 2 3 4
m = d_pool.size
clrs = ['red', 'yellow', 'blue', 'black']  # 颜色
line_width = 3

plt.figure(figsize=(12, 6), facecolor='w')
for i, d in enumerate(d_pool):
    plt.subplot(N - 1, 1, i + 1)
    plt.plot(t, Y_test, 'r-', label=u'真实值', ms=10, zorder=N)
    model.set_params(Poly__degree=d)  # 设置多项式的阶乘
    model.fit(X_train, Y_train)
    lin = model.get_params('Linear')['Linear']
    output = u'%d阶，系数为：' % d
    # LinearRegression将方程分为两个部分存放，coef_存放回归系数，intercept_则存放截距
    print(output, lin.coef_.ravel())

    y_hat = model.predict(X_test)
    s = model.score(X_test, Y_test)

    z = N - 1 if (d == 2) else 0
    label = u'%d阶, 准确率=%.3f' % (d, s)
    plt.plot(t, y_hat, color=clrs[i], lw=line_width, alpha=0.75, label=label, zorder=z)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.ylabel(u'%d阶结果' % d, fontsize=12)

# 预测值和实际值画图比较
plt.legend(loc='lower right')
plt.suptitle(u"线性回归预测时间和功率之间的多项式关系", fontsize=20)
plt.grid(b=True)
plt.show()

# 关于线性回归系数的问题
# 举例：y=a1x1+a2x2+b 它的系数为a1,a2，截距为b
# 1阶时，如果只有1个特征，有一个系数x1，2个特征有3个系数x1、x2 、x1x2，三个特征7个系数 x1、x2、x3、x1x2、x1x3、x2x3、x1x2x3
# 而上面示例中，执行 X_train = ss.fit_transform(X_train) 后，打印X_train, 发现特征只有三个，所以最终打印结果一阶系数有7个