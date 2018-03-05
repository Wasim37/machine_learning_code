import tensorflow as tf
# 引入手写数字库
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# 1.数据加载
# one_hot编码格式的标签数据
mnist = read_data_sets('data/', one_hot=True)
# 看一下数据的维度。。。。。，看一下某一行数据长什么
# 提取训练集的images和label
train_images = mnist.train.images
train_labels = mnist.train.labels
# 01000000000

# 2.数据预处理
# 针对手写数字而言，我们不需要进行预处理

# 3.设置一些超参数
n_inputs = 784  # 28*28
# 分类类别大小
classes = 10

# 学习率
learning_rate = 1e-8
# 批处理数据量
batch_size = 128
# 迭代次数
for_num = 10000

# 数据占位符设置
input_images_x = tf.placeholder(tf.float32, [None, n_inputs])
out_classes_y = tf.placeholder(tf.float32, [None, classes])

# 权重和偏执量
# 网络设计很有关系
w = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 20], stddev=0.1)),
    'wc2': tf.Variable(tf.random_normal([5, 5, 20, 50], stddev=0.1)),
    'wf1': tf.Variable(tf.random_normal([7 * 7 * 50, 500], stddev=0.1)),
    'wf2': tf.Variable(tf.random_normal([500, classes], stddev=0.1))
}
b = {
    'bc1': tf.Variable(tf.random_normal([20], stddev=0.1)),
    'bc2': tf.Variable(tf.random_normal([50], stddev=0.1)),
    'bf1': tf.Variable(tf.random_normal([500], stddev=0.1)),
    'bf2': tf.Variable(tf.random_normal([classes], stddev=0.1))
}


# 4.设计网络
def CNN_LeNet(input_x, w, b):
    # input_x传入的数据是二维的，不满足卷积计算的形式，卷积计算是四维，也就是说要进行reshape
    image_x = tf.reshape(input_x, [-1, 28, 28, 1])

    # conv-1
    # tf.nn.conv2d(input,filter,strides,padding,use_cudnn_on_gpu,name)
    # input 进行卷积计算的数据
    # filter 窗口大小，也是四维度的
    # strides 步长，四个维度的每次走多少远
    # padding 只能是‘SAME’和‘VALID’，‘SAME’表示停留在数据的边缘，‘VALID’表示会做边缘填充
    # use_cudnn_on_gpu 是否使用cudnn加速，默认为True
    # name 变量名
    conv_1 = tf.nn.conv2d(input=image_x, filter=w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    # 激励,激励的是一个线性的方程
    # bias_add 说白就是矩阵加的操作，只是他实现的是，bias加到value上，但是value的维度和bias的不一样
    relu_1 = tf.nn.relu(tf.nn.bias_add(conv_1, b['bc1']))
    # 池化
    pool_1 = tf.nn.max_pool(relu_1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    # 一般为了防止过拟合，我们会加上dropout操作，也可以做标准化，也可以做正则项处理。。。。
    # keep_prob 保留率
    conv_1_out = tf.nn.dropout(pool_1, keep_prob=0.5)

    # conv_2
    conv_2 = tf.nn.conv2d(input=conv_1_out, filter=w['wc2'], strides=[1, 2, 2, 1], padding='SAME')
    # 激励,激励的是一个线性的方程
    # bias_add 说白就是矩阵加的操作，只是他实现的是，bias加到value上，但是value的维度和bias的不一样
    relu_2 = tf.nn.relu(tf.nn.bias_add(conv_2, b['bc2']))
    # 池化
    pool_2 = tf.nn.max_pool(relu_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 一般为了防止过拟合，我们会加上dropout操作，也可以做标准化，也可以做正则项处理。。。。
    # keep_prob 保留率
    conv_2_out = tf.nn.dropout(pool_2, keep_prob=0.5)

    # fc层，全连接层
    # 1.拉伸数据
    wf1_shape = w['wf1'].get_shape().as_list()[0]
    _densel = tf.reshape(conv_2_out, [-1, wf1_shape])
    # 2.执行fc操作
    # 激励，线性方程，w*x+b
    fc1_relu = tf.nn.relu(tf.nn.bias_add(tf.matmul(_densel, w['wf1']), b['bf1']))
    # 3.dropout
    fc1_out = tf.nn.dropout(fc1_relu, keep_prob=0.5)

    # 输出层
    out = tf.nn.bias_add(tf.matmul(fc1_out, w['wf2']), b['bf2'])

    return out

    # 两层卷积层，1层全连接,输出层
    # 输出的数据是28*28*1
    # 输出的数据是10*1
    # 卷积核的大小5*5


# 5.调用网络，设置损失函数，设置优化器，准确率
# （1）建立tensorflow的会话
sess = tf.Session()
#  (2)调用网络
_pred = CNN_LeNet(input_images_x, w, b)
#  (3)设置损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_pred, labels=out_classes_y))
#  (4)设置优化器
# Adam算法，通过栋梁来改善传统梯度下降算法，促进超参数动态调整，引入二次方梯度校正
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#  (5)准确率
# a. 判断预测值和真实值相同有多少
_corr = tf.equal(tf.argmax(_pred, 1), tf.argmax(out_classes_y, 1))
# b.计算true所占的比率
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))

# 6.使用tensorflow做迭代训练
# a.全局变量初始化
init = tf.global_variables_initializer()
sess.run(init)

for i in range(for_num):
    # 计算一下一共需要取多少次数据
    total_batch = int(mnist.train.num_examples / batch_size)
    avg_cost = 0  # 平均误差
    for j in range(total_batch):
        # 取得数据，进行训练
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        loss = sess.run(cost, feed_dict={input_images_x: batch_x, out_classes_y: batch_y})
        avg_cost += loss
    # 每过10次打印一下准确率
    if i % 10 == 0:
        print(sess.run(accr, feed_dict={input_images_x: batch_x, out_classes_y: batch_y}))

        # 7.模型固定，预测，。。。
