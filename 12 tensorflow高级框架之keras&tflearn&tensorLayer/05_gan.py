import tflearn
import tensorflow as tf
import numpy as np
from tflearn.datasets import mnist
X,Y,X_test,Y_test=mnist.load_data(one_hot=True)

image_dim=X.shape[1]
z_dim=200  # 噪音的大小
#total_samples=X.shape[0]
total_samples=len(X)

# 生成模型Generater
def generator(x,reuse=False):
    with tf.variable_scope('Generater',reuse=reuse):
        x=tflearn.fully_connected(x,256,activation='relu')
        x = tflearn.fully_connected(x, image_dim, activation='sigmoid')
        return x

# 判别模型Discriminator
def discriminator(x,reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        x=tflearn.fully_connected(x,256,activation='relu')
        x = tflearn.fully_connected(x, 1, activation='sigmoid')
        return x

# 生成出模型结构
gen_input=tflearn.input_data(shape=[None,z_dim],name='input_noise')
disc_input=tflearn.input_data(shape=[None,image_dim],name='disc_input')

gen_sample=generator(gen_input)
disc_real=discriminator(disc_input)
disc_fake=discriminator(gen_sample,reuse=True)

# 定义损失函数
disc_loss=-tf.reduce_mean(tf.log(disc_real)+tf.log(1.-disc_fake))
gen_loss=-tf.reduce_mean(tf.log(disc_fake))

# 建立训练的优化器并且载入自定义的损失函数
# get_layer_variables_by_scope 得到相关的权重值
gen_vars=tflearn.get_layer_variables_by_scope('Generater')
disc_vars=tflearn.get_layer_variables_by_scope('Discriminator')

gen_model=tflearn.regression(gen_sample,placeholder=None,optimizer='adam',loss=gen_loss,trainable_vars=gen_vars,batch_size=64,name='target_gen',op_name='GEN')
disc_model=tflearn.regression(disc_real,placeholder=None,optimizer='adam',loss=disc_loss,trainable_vars=disc_vars,batch_size=64,name='target_disc',op_name='DISC')

# 训练
gen=tflearn.DNN(gen_model,checkpoint_path='./model/gan/model_gan',tensorboard_dir='./logs')

# 生成模型传入的是噪音，那么我们就需要构建一个噪音数据
z=np.random.uniform(-1.,1.,size=[total_samples,z_dim])

# 数据传入进行训练
gen.fit(X_inputs={gen_input:z,disc_input:X},Y_targets=None,n_epoch=100,run_id='gan_mnist')

# 可视化模型训练效果
import matplotlib.pyplot as plt
f,axis=plt.subplot(10,2,figsize=(10,4))
for i in range(10):
    for j in range(2):
        z = np.random.uniform(-1., 1., size=[1, z_dim])
        temp=[[a,a,a] for a in list(gen.predict([z][0]))]
        axis[i][j].imshow(np.reshape(temp,(28,28,3)))
f.show()
plt.draw()
