import os
import gym
import numpy as np
import torch
from torch import nn
class Agent(nn.Module):
    def __init__(self,algorithm=None,act_dim=0,e_greed=0.1,e_greed_decrement=0):
        assert isinstance(act_dim,int)
        self.act_dim = act_dim

        self.alg = algorithm
        self.global_step = 0
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def sample(self, obs):
        """ 根据观测值 obs 采样（带探索）一个动作"""
        sample = np.random.random()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):
        """ 根据观测值 obs 选择最优动作
        """
        obs = torch.tensor(obs, dtype=torch.float32)
        pred_q = self.alg.predict(obs)
        # print(type(pred_q),pred_q)

        act = pred_q.argmax().numpy() # 选择Q最大的下标，即对应的动作
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        """ 根据训练数据更新一次模型参数"""
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        terminal = np.expand_dims(terminal, axis=-1)

        obs = torch.tensor(obs, dtype=torch.float32)
        act = torch.tensor(act, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        terminal = torch.tensor(terminal, dtype=torch.float32)
        loss = self.alg.learn(obs, act, reward, next_obs, terminal)  # 训练一次网络
        # print(loss.shape,type(loss),'\n',loss)
        return loss.item()

    def save(self,save_path=None,i=0):
        if not os.path.exists('../module'):
            os.makedirs('../module')
        torch.save(self.alg.target_model.state_dict(),f"../module/DQN_{i}.pth")
