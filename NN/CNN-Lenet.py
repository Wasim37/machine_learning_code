
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data


# In[2]:


mnist = input_data.read_data_sets('data/', one_hot=True)
trainimg   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels
print ("MNIST ready")


# In[3]:


#28*28 图片，
n_input  = 784
#输出的大小
n_output = 10  # one_hot  5    0000010000
# 权重
# 
weights  = {
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 64], stddev=0.1)),
        'wc2': tf.Variable(tf.random_normal([5, 5, 64, 128], stddev=0.1)),
        'wd1': tf.Variable(tf.random_normal([7*7*128, 1024], stddev=0.1)),
        'wd2': tf.Variable(tf.random_normal([1024, n_output], stddev=0.1))
    }
# 表数字 i 类的偏置量
biases   = {'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
                      'bc2': tf.Variable(tf.random_normal([128], stddev=0.1)),
                      'bd1': tf.Variable(tf.random_normal([1024], stddev=0.1)),
                      'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1))}


# In[4]:


def conv_basic(_input, _w, _b, _keepratio):
        # [55000,784]
        # INPUT，转换矩阵形状，改成一个28*28*1的，厚度自动
        _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
        # CONV LAYER 1
        #tf.nn.conv2d是TensorFlow里面实现卷积的函数
        #tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
        #除去name参数用以指定该操作的name，与方法有关的一共五个参数：
        #第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一
        #第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
        #第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
        #第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，当其为‘SAME’时，表示卷积核可以停留在图像边缘
        #第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
        #结果返回一个Tensor，这个输出，就是我们常说的feature map
        _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
        #tf.nn.relu：修正线性，max(features, 0)
        #tf.nn.bias_add:这个函数的作用是将偏差项 bias 加到 value 上面。
        #这个操作你可以看做是 tf.add 的一个特例，其中 bias 必须是一维的。
        #该API支持广播形式，因此 value 可以有任何维度。
        #但是，该API又不像 tf.add ，可以让 bias 的维度和 value 的最后一维不同。
        _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
        #最大池化
        #value: 一个四维的Tensor。数据维度是 [batch, height, width, channels]。数据类型是float32，float64，qint8，quint8，qint32。
        #ksize: 一个长度不小于4的整型数组。每一位上面的值对应于输入数据张量中每一维的窗口对应值。
        #strides: 一个长度不小于4的整型数组。该参数指定滑动窗口在输入数据张量每一维上面的步长。
        #padding: 一个字符串，取值为 SAME 或者 VALID 。
        #name: （可选）为这个操作取一个名字。
        _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #处理过拟合操作
        _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)
        
        # CONV LAYER 2
        _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
        _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
        _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)
        
        
        # VECTORIZE 向量化
        _dense1 = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])
        
        # FULLY CONNECTED LAYER 1
        _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
        _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
        # FULLY CONNECTED LAYER 2
        _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
        # RETURN
        out = { 'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool1_dr1': _pool_dr1,
            'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'dense1': _dense1,
            'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out
        }
        return out
print ("CNN READY")


# In[5]:


#tf.random_normal,给出均值为mean，标准差为stdev的高斯随机数（场）
a = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1))
print (a)
a = tf.Print(a, [a], "a: ")
#Variable的初始化
init = tf.global_variables_initializer()
#建立会话
sess = tf.Session()
#执行初始化
sess.run(init)
sess.run(a)


# In[6]:


#print (help(tf.nn.conv2d))
# print (help(tf.nn.max_pool))


# In[ ]:


#通过操作符号变量来描述这些可交互的操作单元
#x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。
#我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。
#我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。（这里的None表示此张量的第一个维度可以是任何长度的。）
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

# FUNCTIONS

#调用CNN函数，返回运算完的结果
_pred = conv_basic(x, weights, biases, keepratio)['out']
#交叉熵
#首先看输入logits，它的shape是[batch_size, num_classes] ，
#一般来讲，就是神经网络最后一层的输出z。
#另外一个输入是labels，它的shape也是[batch_size, num_classes]，就是我们神经网络期望的输出。
#这个函数的作用就是计算最后一层是softmax层的cross entropy，只不过tensorflow把softmax计算与cross entropy计算放到一起了。
#用一个函数来实现，用来提高程序的运行速度
#http://www.jianshu.com/p/fb119d0ff6a6
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=_pred,labels=y))
#Adam算法
#。AdamOptimizer通过使用动量（参数的移动平均数）来改善传统梯度下降，促进超参数动态调整。
#我们可以通过创建标签错误率的摘要标量来跟踪丢失和错误率
#一个寻找全局最优点的优化算法，引入了二次方梯度校正。
#相比于基础SGD算法，1.不容易陷于局部优点。2.速度更快
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
#比较
_corr = tf.equal(tf.argmax(_pred,1), tf.argmax(y,1)) 
#cast:将x或者x.values转换为dtype
#tf.reduce_mean  求tensor中平均值
#http://blog.csdn.net/lenbow/article/details/52152766
accr = tf.reduce_mean(tf.cast(_corr, tf.float32)) 
# 初始化
init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer() 
# SAVER
print ("GRAPH READY")


# In[ ]:


sess = tf.Session()
sess.run(init)
sess.run(init_local)
#训练次数
training_epochs = 15
#batch  根据神经网络中间最大维度的那个矩阵大小所占内存空间，来确定batch_size的最大值
# [batch_size,hight,width,depth]
# [batch_size,32,32,1024]
# (1024*32*32/1024/1024/1024)*4 代表一个int型训练数据的矩阵所占空间（G）
# (1024*32*32/1024/1024/1024)*8 代表一个float型训练数据的矩阵所占空间（G）
# 并且根据内存的一半作为最大空间，来换算batch_size的值
# batch_size 一般是2的次方
# 这个数值还和 样本总量有关
batch_size      = 8
#执行到第几次显示运行结果
display_step    = 10
for epoch in range(training_epochs):
    #平均误差
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    #total_batch = 10
    # Loop over all batches  循环所有批次
    list1=[]
    for i in range(total_batch):
        #去除训练集合的下10条
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data 使用批处理数据进行培训
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keepratio:0.7})
        # Compute average loss  计算平均损失
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})/total_batch
        train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})
        
    # Display logs per epoch step 显示现在的状态
    if epoch % display_step == 0: 
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})
        
        print (" Training accuracy: %.3f" % (train_acc))
        test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel, keepratio:1.})
        print (" Test accuracy: %.3f" % (test_acc))

print ("OPTIMIZATION FINISHED")


# In[ ]:


#1.数据载入+预处理
#2.神经网络结构设计
#3.选择损失函数
#4.选择SGD优化函数、设置正确率计算方法
#5.设置迭代次数、每次训练的数据量
#6.执行迭代训练（设置每多少次，看一下当前状态）

