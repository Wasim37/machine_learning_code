# coding=UTF-8
'''
Created on 2017年6月25日

@author: Administrator
'''
# 二元隐马尔科夫模型（Bigram HMMs）
# 'trainCorpus.txt_utf8'为人民日报已经人工分词的预料，29万多条句子

import sys

#state_M = 4
#word_N = 0
A_dic = {}
B_dic = {}
Count_dic = {}
Pi_dic = {}
word_set = set()
state_list = ['B','M','E','S']
line_num = -1

INPUT_DATA = "trainCorpus.txt_utf8"
PROB_START = "trainHMM\prob_start.py"   #初始状态概率
PROB_EMIT = "trainHMM\prob_emit.py"     #发射概率
PROB_TRANS = "trainHMM\prob_trans.py"   #转移概率


def init():  #初始化字典，初始值为0
    for state in state_list:
        A_dic[state] = {}
        for state1 in state_list:
            A_dic[state][state1] = 0.0
    for state in state_list:
        Pi_dic[state] = 0.0
        B_dic[state] = {}
        Count_dic[state] = 0


def getList(input_str):  #输入词语，输出状态
    outpout_str = []
    if len(input_str) == 1:
        outpout_str.append('S')
    elif len(input_str) == 2:
        outpout_str = ['B','E']
    else:
        M_num = len(input_str) -2
        M_list = ['M'] * M_num
        outpout_str.append('B')
        outpout_str.extend(M_list)  
        outpout_str.append('E')
    return outpout_str


def Output():   #输出模型的三个参数：初始概率+转移概率+发射概率
    start_fp = open(PROB_START,'w')
    emit_fp = open(PROB_EMIT,'w')
    trans_fp = open(PROB_TRANS,'w')
    print ("len(word_set) = %s " % (len(word_set)))

    for key in Pi_dic:      #状态的初始概率
        Pi_dic[key] = Pi_dic[key] * 1.0 / line_num
    print >>start_fp,Pi_dic

    for key in A_dic:      #状态转移概率
        for key1 in A_dic[key]:
            A_dic[key][key1] = A_dic[key][key1] / Count_dic[key]
    print >>trans_fp,A_dic

    for key in B_dic:        #发射概率
        for word in B_dic[key]:
            B_dic[key][word] = B_dic[key][word] / Count_dic[key]
    print >>emit_fp,B_dic

    start_fp.close()
    emit_fp.close()
    trans_fp.close()


def main():

    ifp = open(INPUT_DATA)
    init()
    global word_set   #初始是set()，定义全局变量，这个main函数里都起作用
    global line_num   #初始是-1
    for line in ifp:
        line_num += 1
        if line_num % 10000 == 0:
            print (line_num)
        line = line.strip()  #去掉前后空格
        if not line:continue
      
        line = line.decode("utf-8","ignore")  #解码


        word_list = []
        for i in range(len(line)):
            if line[i] == " ":continue
            word_list.append(line[i])#把所有的词放到word_list里
        word_set = word_set | set(word_list)   #训练语料库所有字的集合，不重复的


        lineArr = line.split(" ")  #以空格为分割，转化为一个数组
        line_state = []
        for item in lineArr:
            line_state.extend(getList(item))  #把得到的集合加入到line_state里面
        if len(word_list) != len(line_state):
            print >> sys.stderr,"[line_num = %d][line = %s]" % (line_num, line.endoce("utf-8",'ignore'))#输出错误
        else:
            for i in range(len(line_state)):
                if i == 0:
                    Pi_dic[line_state[0]] += 1   #用于计算初始状态概率
                    Count_dic[line_state[0]] += 1   #记录每一个状态的出现的概率
                else:
                    A_dic[line_state[i-1]][line_state[i]] += 1    #用于计算转移概率
                    Count_dic[line_state[i]] += 1
                    if not B_dic[line_state[i]].has_key(word_list[i]):
                        B_dic[line_state[i]][word_list[i]] = 0.0
                    else:
                        B_dic[line_state[i]][word_list[i]] += 1   #用于计算发射概率
                        
    Output()
    ifp.close()


if __name__ == "__main__":
    main()