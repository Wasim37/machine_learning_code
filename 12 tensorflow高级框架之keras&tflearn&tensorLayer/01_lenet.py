import tflearn.datasets.mnist as mnist #引入数据
# 引入模型相关的方法
from tflearn.layers.core import input_data,dropout,fully_connected  #数据操作常用模块
from tflearn.layers.conv import conv_2d,max_pool_2d # cnn层级结构模块
from tflearn.layers.normalization import local_response_normalization # 进行神经元的抑制
import tensorflow as tf
import tflearn

#
from tflearn.layers.estimator import regression

X,Y,X_test,Y_test=mnist.load_data(one_hot=True)
#print(X.shape) #55000,784
# 55000,28,28,1 需要这样形式的数据
X=X.reshape([-1,28,28,1])
X_test=X_test.reshape([-1,28,28,1])


with tf.name_scope('lenet'):
    # 构建网络
    # 第一不需要建立w和b这两个变量
    network=input_data(shape=[None,28,28,1],name='input')
    network=conv_2d(network,filter_size=5,nb_filter=64,activation='relu',regularizer='L2')
    # filter_size 卷积核的大小
    # nb_filter 有多少个filter
    # activation 激活函数
    # regularizer 正则化
    network=max_pool_2d(network,kernel_size=2,strides=2)
    # kernel_size，池化的大小

    # 局部归一化
    # 仿造生物学上活跃的神经元与对相邻神经元的抑制现象（侧抑制）
    network=local_response_normalization(network)


    network=conv_2d(network,filter_size=5,nb_filter=128,activation='relu',regularizer='L2')
    network=max_pool_2d(network,kernel_size=2,strides=2)
    network=local_response_normalization(network)

    # 全连接层
    network=fully_connected(network,1024,activation='tanh')
    network=dropout(network,keep_prob=0.5)

    network=fully_connected(network,10,activation='softmax')

    # 损失函数以及优化器
    network=regression(network,optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001,name='target')

# 训练
with tf.name_scope('train'):
    # 启动网络（实例化）
    # checkpoint_path 持久化模型存储地址
    # tensorboard_dir tensorboard日志路径
    # max_checkpoints 最大存储模型的个数
    # tensorboard_verbose 日志详细等级
    # --0: loss,accuracy
    # --1: loss,accuracy,Gradients
    # --2: loss,accuracy,Gradients,weights
    # --3: loss,accuracy,Gradients,weights,Activations,Sparsity
    model=tflearn.DNN(network,checkpoint_path='./model/lenet/model',
                      tensorboard_dir='./logs',max_checkpoints=1,tensorboard_verbose=3)

    # validation_set 接收一个元祖或者数值
    # n_epoch 迭代次数
    # shuffle 打乱
    # snapshot_epoch  bool，如果True，那么每次迭代结束，都会评估一下模型
    # snapshot_step   迭代多少次保存一下模型
    # run_id 可视化网络名字
    model.fit(X,Y,validation_set=(X_test,Y_test),n_epoch=5,shuffle=True,snapshot_epoch=True,snapshot_step=100,batch_size=1024,run_id='lenet')