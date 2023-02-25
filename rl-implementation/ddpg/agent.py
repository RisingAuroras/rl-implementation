import torch
import logging
from torch import nn
import numpy as np



class Agent():
    def __init__(self, algorithm, act_dim, expl_noise=0.1):
        assert isinstance(act_dim, int)
        self.alg = algorithm

        self.act_dim = act_dim
        self.expl_noise = expl_noise

        # 注意：最开始先同步self.model和self.target_model的参数.
        self.alg.sync_target(decay=0)

    def sample(self, obs):
        """ 根据观测值 obs 采样（带探索）一个动作
        """
        action = self.predict(torch.tensor(obs,dtype=torch.float32)).item()
        action_noise = torch.normal(mean=0., std=self.expl_noise, size=(self.act_dim,))
        action = (action + action_noise).clip(-1, 1)
        return action

    def predict(self, obs):
        """ 根据观测值 obs 选择最优动作
        """
        obs = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
        action = self.alg.predict(obs)
        action = torch.clip(action,-1.,1.)
        return action

    def learn(self, obs, action, reward, next_obs, terminal):
        """ 根据训练数据更新一次模型参数
        """
        terminal = np.expand_dims(terminal, -1)# 在最后加一维 （a,b，..., c) -> (a,b,...,c,1)
        reward = np.expand_dims(reward, -1)# 在最后加一维

        obs = torch.tensor(obs, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        terminal = torch.tensor(terminal, dtype=torch.float32)
        critic_loss, actor_loss = self.alg.learn(obs, action, reward, next_obs,terminal)
        return critic_loss, actor_loss
