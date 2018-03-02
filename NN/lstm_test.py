# encoding=utf8
import numpy as np

from lstm import LstmParam, LstmNetwork

class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
        return 1/2*(pred[1] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff

def example_0():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    # parameters for input data dimension and lstm cell count
    # 指定神经元个数为100
    mem_cell_ct = 100
    # 制定输入样本的特征维度数量 输入4个样本，每个样本是1*50的维度
    x_dim = 50
    # 用神经元个数和样本维度数初始化LSTM
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    # 初始化网络
    lstm_net = LstmNetwork(lstm_param)
    # 标签、目标输出
    # 随机输入4个50维的样本，去拟合下面的四个数字。lstm的拟合能力非常强
    y_list = [-0.5, 0.2, 0.1, -0.5]
    # 生成输入数据
    # list推导式：http://blog.chinaunix.net/uid-28631822-id-3488324.html
    input_val_arr = [np.random.random(x_dim) for _ in y_list]
    #for _ in y_list:
        #input_val_arr.append(np.random.random(x_dim))
    for cur_iter in range(1000):
        print("iter", "%2s" % str(cur_iter), end=": ")
        for ind in range(len(y_list)):#[0,1,2,3]
            lstm_net.x_list_add(input_val_arr[ind])

        print("y_pred = [" +
              ", ".join(["% 2.5f" % lstm_net.lstm_node_list[ind].state.h[0] for ind in range(len(y_list))]) +
              "]", end=", ")

        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        print("loss:", "%.3e" % loss)
        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()


if __name__ == "__main__":
    example_0()
