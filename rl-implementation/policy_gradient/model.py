import numpy
import torch
from torch import nn
from torch import functional as F

class Model(nn.Module):

    """
    使用全连接网络,离散问题
    参数:
        obs_dim(int):观测空间的维度
        act_dim(int):动作空间的维度
    """


    def __init__(self,obs_dim,act_dim):
        super(Model, self).__init__()
        hid1_size = 64
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, act_dim)


    def forward(self, x):  # 可直接用 model = Model(5); model(obs)调用
        out = torch.tanh(self.fc1(x))# out's shape is 20
        prob = torch.softmax(self.fc2(out), axis=-1) # prob's shape is 2,通过softmax换算成做出动作的概率，即向左向右两个动作的概率
        return prob
