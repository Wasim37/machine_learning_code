# encoding=utf8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage.io import imsave
import os
import shutil

img_height = 28
img_width = 28
img_size = img_height * img_width

to_train = False
to_restore = False
output_path = "output"

# 总迭代次数500
max_epoch = 801

h1_size = 150
h2_size = 300
z_size = 100
batch_size = 256


# generate (model 1)
def build_generator(z_prior):
    # 网络类型与网络层数随便定，只要能收敛
    # 第一层神经网络
    # tf.truncated_normal产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。
    w1 = tf.Variable(tf.truncated_normal([z_size, h1_size], stddev=0.1), name="g_w1", dtype=tf.float32)
    # 产生和h1_size变量一样维度的0
    b1 = tf.Variable(tf.zeros([h1_size]), name="g_b1", dtype=tf.float32)
    # z_prior维度为TensorShape([Dimension(256), Dimension(100)])，w1维度为TensorShape([Dimension(100), Dimension(150)])
    # tf.matmul(z_prior, w1)相乘以后维度为【256,150】，b维度为TensorShape([Dimension(150)])
    h1 = tf.nn.relu(tf.matmul(z_prior, w1) + b1)

    # 第二层神经网络
    w2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.1), name="g_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h2_size]), name="g_b2", dtype=tf.float32)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    # 第三层神经网络
    w3 = tf.Variable(tf.truncated_normal([h2_size, img_size], stddev=0.1), name="g_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([img_size]), name="g_b3", dtype=tf.float32)
    h3 = tf.matmul(h2, w3) + b3

    # x_generate为产生的图片矩阵，它和h3维度一样
    x_generate = tf.nn.tanh(h3)
    # g_params为生成模型所有权重和biase值
    g_params = [w1, b1, w2, b2, w3, b3]
    return x_generate, g_params


# discriminator (model 2)
def build_discriminator(x_data, x_generated, keep_prob):
    # tf.concat
    # t1 = [[1, 2, 3], [4, 5, 6]]
    # t2 = [[7, 8, 9], [10, 11, 12]]
    # tf.concat(0, [t1, t2]) == > [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    x_in = tf.concat([x_data, x_generated], 0)
    # 判别器第一层神经网络
    w1 = tf.Variable(tf.truncated_normal([img_size, h2_size], stddev=0.1), name="d_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h2_size]), name="d_b1", dtype=tf.float32)
    h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in, w1) + b1), keep_prob)
    # 判别器第二层神经网络
    w2 = tf.Variable(tf.truncated_normal([h2_size, h1_size], stddev=0.1), name="d_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h1_size]), name="d_b2", dtype=tf.float32)
    h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)
    # 判别器第三层神经网络
    w3 = tf.Variable(tf.truncated_normal([h1_size, 1], stddev=0.1), name="d_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([1]), name="d_b3", dtype=tf.float32)
    h3 = tf.matmul(h2, w3) + b3
    # [0, 0]是起点位置
    # [batch_size, -1]切出多大，即行数和列数,-1表示全部列数
    y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [batch_size, -1], name=None))
    print(y_data.eval)
    y_generated = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1, -1], name=None))
    d_params = [w1, b1, w2, b2, w3, b3]
    # y_data真实样本放入判别器的输出，y_generated生成器生成样本放入判别器的输出
    return y_data, y_generated, d_params


#
def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], img_height, img_width)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)


def train():
    # load data（mnist手写数据集）
    mnist = input_data.read_data_sets('F:\\\\', one_hot=True)

    x_data = tf.placeholder(tf.float32, [batch_size, img_size], name="x_data")
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # 创建生成模型
    x_generated, g_params = build_generator(z_prior)
    # 创建判别模型
    y_data, y_generated, d_params = build_discriminator(x_data, x_generated, keep_prob)

    # 损失函数的设置
    d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))
    g_loss = - tf.log(y_generated)

    optimizer = tf.train.AdamOptimizer(0.0001)

    # 两个模型的优化函数
    d_trainer = optimizer.minimize(d_loss, var_list=d_params)
    g_trainer = optimizer.minimize(g_loss, var_list=g_params)

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()
    # 启动默认图
    sess = tf.Session()
    # 初始化
    sess.run(init)

    if to_restore:
        chkpt_fname = tf.train.latest_checkpoint(output_path)
        saver.restore(sess, chkpt_fname)
    else:
        if not os.path.exists(output_path):
            os.mkdir(output_path)

    # z_sample_val产生随机噪音
    z_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)

    steps = 60000 / batch_size
    for i in range(sess.run(global_step), max_epoch):

        for j in np.arange(steps):
            #         for j in range(steps):
            print("epoch:%s, iter:%s" % (i, j))
            # 每一步迭代，我们都会加载256个训练样本，然后执行一次train_step
            x_value, _ = mnist.train.next_batch(batch_size)
            x_value = 2 * x_value.astype(np.float32) - 1
            # z_value产生随机噪音
            z_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
            # 执行生成
            # sess.run([y_data,y_generated,d_trainer],
            # feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
            a, b, c = sess.run([y_data, y_generated, d_trainer],
                               feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.5).astype(np.float32)})
            # .run(fetches)
            # 执行判别
            if j % 2 == 0:
                sess.run(g_trainer,
                         feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
        # x_generated为生成器模型
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val})

        z_random_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_random_sample_val})
        # show_result(x_gen_val, "output/random_sample{0}.jpg".format(i))
        sess.run(tf.assign(global_step, i + 1))
        if i % 400 == 0:
            show_result(x_gen_val, "output/sample{0}.jpg".format(i))
            show_result(x_gen_val, "output/random_sample{0}.jpg".format(i))
            saver.save(sess, os.path.join(output_path, "model"), global_step=global_step)


def test():
    # 定义一个随机噪声的形状
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    # 定义生成网络
    x_generated, _ = build_generator(z_prior)
    # 检查模型文件output_path="output"
    chkpt_fname = tf.train.latest_checkpoint(output_path)
    # 初始化参数函数
    init = tf.initialize_all_variables()
    # 开启一个会话
    sess = tf.Session()
    # 定义读取模型文件方法
    saver = tf.train.Saver()
    # 开始运行初始化方法
    sess.run(init)
    # 读入模型文件
    saver.restore(sess, chkpt_fname)
    # 用生成网络生成图像
    for i in range(100):
        # 随机生成噪声数据
        z_test_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
        # 输入随机生成噪声数据给生成网络
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_test_value})
        # 把生成网络的输出显示为图片
        show_result(x_gen_val, "output_img/test_result{0}.jpg".format(i))


if __name__ == '__main__':
    if 0:
        train()
    else:
        test()
