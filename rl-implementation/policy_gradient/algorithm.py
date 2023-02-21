
import torch
from torch import nn
import torch.nn.functional as F
class PolicyGradient():
    def __init__(self, model, lr):
        """ Policy Gradient algorithm

        Args:
            model (Model): policy的前向网络.
            lr (float): 学习率.
        """
        assert isinstance(lr, float)

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)

    def predict(self, obs):
        """ 使用policy model预测输出的动作概率
        """
        prob = self.model(obs)
        # print(prob,type(prob))
        return prob
    def learn(self, obs, action, reward):
        """ 用policy gradient 算法更新policy model
        """
        prob = self.model(obs)  # 获取输出动作概率
        # log_prob = Categorical(prob).log_prob(action) # 交叉熵
        # loss = paddle.mean(-1 * log_prob * reward)
        action_onehot = torch.squeeze(
            F.one_hot(action.to(torch.int64), num_classes=prob.shape[1]), axis=1)
        log_prob = torch.sum(torch.log(prob) * action_onehot, axis=-1)
        reward = torch.squeeze(reward, axis=1)
        loss = torch.mean(-1. * log_prob * reward)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss