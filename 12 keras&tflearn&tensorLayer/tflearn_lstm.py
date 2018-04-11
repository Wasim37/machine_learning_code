# 电影评论及评论对应的情感分类
import tflearn
from tflearn.datasets import imdb # IMDB 是一个电影评论的数据库
from tflearn.data_utils import to_categorical,pad_sequences

train, valid, _=imdb.load_data()

X,Y=train
X_test,Y_test=valid

# 序列处理
# 1.填充序列
# pad_sequences(sequences,maxlen,dtype,padding,truncating,value)
# 将长度为nb_samples的序列（标量序列）转换为形式如同（nb_samples，nb_timesteps） 2d 矩阵。
# 如果提供了参数maxlen，那么nb_timesteps=maxlen，
# 否则则nb_timesteps为最长序列的长度
# 其他短于nb_timesteps的序列，后面部分都会用value填充
# 其他长于nb_timesteps的序列，后面部分都会被截取
# 截取的位置取决于padding和truncating
train_X=pad_sequences(X,maxlen=100,value=0.)
test_X=pad_sequences(X_test,maxlen=100,value=0.)

# 2.标签数据进行二进制矩阵化
train_Y=to_categorical(Y,2)
test_Y=to_categorical(Y_test,2)

# 构建网络
net=tflearn.input_data([None,100])
# embedding处理，因为外部数据没有word2vec
# 输入维度是词语的空间，嵌入到一个128的向量空间
net=tflearn.embedding(net,input_dim=100000,output_dim=128)
# lstm
net=tflearn.lstm(net,128,dropout=0.8)
# 全连接
net=tflearn.fully_connected(net,2,activation='softmax')
# 设置损失和优化器
net=tflearn.regression(net,optimizer='adam',
               loss='categorical_crossentropy',learning_rate=0.001)

# 训练
model=tflearn.DNN(net,tensorboard_dir='./logs',checkpoint_path='./model/lstm/model_lstm')
model.fit(train_X,train_Y,validation_set=(test_X,test_Y),run_id='lstm',batch_size=32)


