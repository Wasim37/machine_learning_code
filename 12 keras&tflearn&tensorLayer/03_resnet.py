import tflearn
from tflearn.datasets import cifar10

# cifar10是一个用于普适物体识别的数据集，由60000张32*32的 RGB 彩色图片构成，共10个分类。
# cifar10 简介：http://blog.csdn.net/zeuseign/article/details/72773342

# ResNet网络详解：https://www.jianshu.com/p/e58437f39f65
# ResNet学习：http://blog.csdn.net/xxy0118/article/details/78324256

# 引入数据
(X,Y),(X_test,Y_test)=cifar10.load_data()
# 对于标签的值进行矩阵二进制化
Y=tflearn.data_utils.to_categorical(Y,10)
Y_test=tflearn.data_utils.to_categorical(Y_test,10)

# 图像处理相关方法
# 1.实时数据预处理
# img_prep=tflearn.data_preprocessing.ImagePreprocessing()
img_prep=tflearn.ImagePreprocessing()
# 零中心，在整个数据集上进行均值计算
img_prep.add_featurewise_zero_center()
# 标准化
img_prep.add_featurewise_stdnorm()

# 2.实时数据添加
img_aug=tflearn.ImageAugmentation()
# 随机左右翻转图像
img_aug.add_random_flip_leftright()
# 随机上下翻转图像
img_aug.add_random_flip_updown()
# 自定义角度随机翻转
img_aug.add_random_rotation(max_angle=25.)

# 构建网络
network=tflearn.input_data(shape=[None,32,32,3],data_augmentation=img_aug,data_preprocessing=img_prep)

network=tflearn.conv_2d(network,16,3,regularizer='l2',weight_decay=0.0001)

# 载入残差结构
n=3
network=tflearn.resnext_block(network,nb_blocks=n,out_channels=16,cardinality=32)
network=tflearn.resnext_block(network,nb_blocks=1,out_channels=32,cardinality=32,downsample=True)# 残差虚线

network=tflearn.resnext_block(network,nb_blocks=n,out_channels=32,cardinality=32)
network=tflearn.resnext_block(network,nb_blocks=1,out_channels=32,cardinality=32,downsample=True)# 残差虚线

network=tflearn.resnext_block(network,nb_blocks=n,out_channels=64,cardinality=32)
network=tflearn.resnext_block(network,nb_blocks=1,out_channels=32,cardinality=32,downsample=True)# 残差虚线

network=tflearn.batch_normalization(network)# 标准化
network=tflearn.activation(network,'relu') # 镶嵌激活函数
network=tflearn.global_avg_pool(network) # 镶嵌平均池化
# 全连接（输出）
network=tflearn.fully_connected(network,10,activation='softmax')
# 设置优化器的参数
# 动态学习率，初始值为0.1,每decay_step时乘以lr_decay
opt=tflearn.Momentum(learning_rate=0.1,lr_decay=0.1,decay_step=32000)
network=tflearn.regression(network,optimizer=opt)

# 训练
model=tflearn.DNN(network,checkpoint_path='./model/resnet/model_resnet',tensorboard_dir='./logs')
model.fit(X,Y,n_epoch=200,validation_set=(X_test,Y_test),batch_size=128)

