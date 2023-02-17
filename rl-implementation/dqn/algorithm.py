import copy
import numpy
import torch
from torch import nn
import torch.nn.functional as F
"""
DQN Algorithm
"""
class DQN():
    def __init__(self,model,gamma=None,lr=None):
        """
        :param model:定义Q函数的前向网络结构
        :param gamma: reward的衰减因子
        :param lr: learning_rate，学习率.
        """
        super(DQN, self).__init__()

        # checks
        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.model = model
        self.target_model = copy.deepcopy(model)

        self.gamma = gamma
        self.lr = lr

        self.mse_loss = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(model.parameters(),lr=self.lr)  # 使用Adam优化器
    def predict(self,obs):
        """
        使用self.model的网络来获取 [Q(s,a1),Q(s,a2)
        :param obs:
        :return:
        """
        return self.model(obs)
    def learn(self,obs,action,reward,next_obs,terminal):
        '''使用DQN算法更新self.model的value网络'''
        pred_values = self.model(obs)
        action_dim = pred_values.shape[-1]
        # print(f"action's type is {type(action_dim)} and {action_dim.shape} and {action_dim}")
        # print(f"action's type is {type(action)} and {action.shape} and {action}")
        action_onehot =  F.one_hot(action.to(torch.int64),num_classes=action_dim).squeeze()
        # print(action_onehot.shape,pred_values.shape)
        # print(pred_values)
        pred_value = pred_values * action_onehot
        pred_value = pred_value.sum(axis=1,keepdim=True)
        # print(pred_value)

        # 从target_model中获取 max Q' 的值，用于计算target_Q
        with torch.no_grad():
            max_v = self.target_model(next_obs).max(1, keepdim=True)[0]
            target = reward + (1 - terminal) * self.gamma * max_v
        # if True or pred_value.shape != target.shape:
        #     print(pred_value.shape,target.shape)
        #     print(action_onehot.shape)
        #     print(action_onehot)
        #     exit(-1)
        loss =self.mse_loss(pred_value,target)

        # 计算 Q(s,a) 与 target_Q的均方差，得到loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


    def sync_target(self):
        """ 把 self.model 的模型参数值同步到 self.target_model"""
        self.target_model.load_state_dict(self.model.state_dict())

