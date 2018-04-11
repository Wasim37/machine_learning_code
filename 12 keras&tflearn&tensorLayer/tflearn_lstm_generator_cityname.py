# 使用lstm学习城市名称，并且进行数据的生成
# 这里只是告诉你怎么进行数据的预测，其他比如语义方面不进行考虑
import tflearn
import requests

# text=requests.get('https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/US_Cities.txt').text
# with open('US_Cities.txt','w') as f:
#     f.write(text)
string_utf8=open('US_Cities.txt').read()

# 生成词典 word2vec
maxlen=20 # 每个序列最长长度
X,Y,char_idx=tflearn.data_utils.string_to_semi_redundant_sequences(string_utf8,seq_maxlen=maxlen,redun_step=3)
print(char_idx)

#print(X.shape)
# string='床前明月光,疑是地上霜。举头望明月，低头思故乡。'
# X,Y,char_idx=tflearn.data_utils.string_to_semi_redundant_sequences(string,seq_maxlen=4,redun_step=2)
# print(X.shape)#(3,5,9)
# print(X)
# # 1：张三\n李四
# # 2：\n李四\n王
# # 3：四\n王五\n
# # 4:王五\n赵六
# # 表示生成3个序列,每个序列为5个字符，一共有9列字典的id
# print(Y.shape)#(3,9)
# # 预测
# # 1:\n
# # 2:四
# # 3:王
# # 4:\n
# print(char_idx)#9个元素的字典

# 构建网络
g=tflearn.input_data(shape=[None,maxlen,len(char_idx)])
g=tflearn.layers.recurrent.lstm(g,512,return_seq=True,name='g1')
g=tflearn.dropout(g,0.5,name='d1')
g=tflearn.layers.recurrent.lstm(g,512,name='g2')
g=tflearn.dropout(g,0.5,name='d2')
# 全连接
g=tflearn.fully_connected(g,len(char_idx),activation='softmax')
g=tflearn.regression(g,optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001)# 设置损失和优化器

# 序列生成的深层神经网络模型
# clip_gradients 梯度
m=tflearn.SequenceGenerator(g,dictionary=char_idx,seq_maxlen=maxlen,clip_gradients=5.0,checkpoint_path='./model/lstm_gen/model',tensorboard_dir='./logs')

# 循环遍历，进行序列生成
for i in range(40):
    # 建立生成序列的种子,随机的
    seed=list(tflearn.data_utils.random_sequence_from_string(string_utf8,maxlen))
    # 填充数据进行训练
    m.fit(X,Y,validation_set=0.1,batch_size=1024,n_epoch=1,run_id='us_cities')
    # 调用模型进行数据生成
    # temperature  新颖程度
    # 0 表示就是样本数据
    print(''.join(m.generate(seq_length=30,temperature=1.5,seq_seed=seed)))
    print(''.join(m.generate(seq_length=30, temperature=1., seq_seed=seed)))
    print(''.join(m.generate(seq_length=30, temperature=.5, seq_seed=seed)))
