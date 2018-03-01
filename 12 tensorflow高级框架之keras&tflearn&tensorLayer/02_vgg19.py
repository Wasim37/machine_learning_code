from tflearn.datasets import oxflower17 #各种花的分类数据
from tflearn.layers.core import dropout,input_data,fully_connected
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.estimator import regression
import tflearn

# oxflower17 数据集没有测试集，可以通过 validation_set 手动分出测试集
X,Y=oxflower17.load_data(one_hot=True,resize_pics=(224,224))

# 建立VGG19结构
# VGG结构详解：http://www.cnblogs.com/vipyoumay/p/7884472.html
# VGG 在特征提取上经常有奇效

network=input_data(shape=[None,224,224,3])
# 第一层
network=conv_2d(network,nb_filter=64,filter_size=3,activation='relu',name='block1_conv1')
network=conv_2d(network,nb_filter=64,filter_size=3,activation='relu',name='block1_conv2')
network=max_pool_2d(network,kernel_size=2,strides=2,name='block1_pool')

# 第二层
network=conv_2d(network,nb_filter=128,filter_size=3,activation='relu',name='block2_conv1')
network=conv_2d(network,nb_filter=128,filter_size=3,activation='relu',name='block2_conv2')
network=max_pool_2d(network,kernel_size=2,strides=2,name='block2_pool')

# 第三层
network=conv_2d(network,nb_filter=256,filter_size=3,activation='relu',name='block3_conv1')
network=conv_2d(network,nb_filter=256,filter_size=3,activation='relu',name='block3_conv2')
network=conv_2d(network,nb_filter=256,filter_size=3,activation='relu',name='block3_conv3')
network=conv_2d(network,nb_filter=256,filter_size=3,activation='relu',name='block3_conv4')
network=max_pool_2d(network,kernel_size=2,strides=2,name='block3_pool')

# 第四层
network=conv_2d(network,nb_filter=512,filter_size=3,activation='relu',name='block4_conv1')
network=conv_2d(network,nb_filter=512,filter_size=3,activation='relu',name='block4_conv2')
network=conv_2d(network,nb_filter=512,filter_size=3,activation='relu',name='block4_conv3')
network=conv_2d(network,nb_filter=512,filter_size=3,activation='relu',name='block4_conv4')
network=max_pool_2d(network,kernel_size=2,strides=2,name='block4_pool')

# 第五层
network=conv_2d(network,nb_filter=512,filter_size=3,activation='relu',name='block5_conv1')
network=conv_2d(network,nb_filter=512,filter_size=3,activation='relu',name='block5_conv2')
network=conv_2d(network,nb_filter=512,filter_size=3,activation='relu',name='block5_conv3')
network=conv_2d(network,nb_filter=512,filter_size=3,activation='relu',name='block5_conv4')
network=max_pool_2d(network,kernel_size=2,strides=2,name='block5_pool')

# 拉平，只是为了方便全连接计算
flatten_layer=tflearn.layers.core.flatten(network,name='flatten')

# 全连接
network=fully_connected(flatten_layer,4096,activation='relu')
network=dropout(network,0.5)
network=fully_connected(network,4096,activation='relu')
network=dropout(network,0.5)
network=fully_connected(network,1000,activation='relu')
network=dropout(network,0.5)
# 输出 最终分出的结果有17类
network=fully_connected(network,17,activation='softmax')
# 建立优化器和损失函数，regression本身有很多默认值
network=regression(network, optimizer='adam',
               loss='categorical_crossentropy',learning_rate=0.001)


# 训练
model=tflearn.DNN(network,checkpoint_path='./model/vgg19/model_vgg',tensorboard_dir='logs')
model.fit(X,Y,validation_set=0.1,shuffle=True,show_metric=True,batch_size=4,snapshot_step=200,snapshot_epoch=False,run_id='vgg')