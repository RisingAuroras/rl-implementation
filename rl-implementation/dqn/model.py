import numpy
import torch
from torch import nn

class Model(nn.Module):

    """
    使用全连接网络
    参数:
        obs_dim(int):观测空间的维度
        act_dim(int):动作空间的维度
    """


    def __init__(self,obs_dim,act_dim):
        super(Model, self).__init__()
        hid1_size  = 128
        hid2_size = 128
        self.model = nn.Sequential(
            nn.Linear(obs_dim,hid1_size),
            nn.ReLU(),
            nn.Linear(hid1_size,hid2_size),
            nn.ReLU(),
            nn.Linear(hid1_size,act_dim)
        )
    def forward(self,obs):
        Q = self.model(obs)
        return Q