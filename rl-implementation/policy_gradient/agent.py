import os
import gym
import numpy as np
import torch
from torch import nn
class Agent():
    def __init__(self,algorithm=None):
        self.alg = algorithm

    def sample(self, obs):
        """ 根据观测值 obs 采样（带探索）一个动作
        """
        obs = torch.tensor(obs, dtype=torch.float32)
        prob = self.alg.predict(obs)
        prob = prob.detach().numpy()
        act = np.random.choice(len(prob), 1, p=prob)[0]  # 根据动作概率选取动作
        return act

    def predict(self, obs):
        """ 根据观测值 obs 选择最优动作
        """
        obs =  torch.tensor(obs, dtype=torch.float32)
        prob = self.alg.predict(obs)
        act = prob.detach().argmax().item() # 根据动作概率选择概率最高的动作
        return act

    def learn(self, obs, act, reward):
        """ 根据训练数据更新一次模型参数
        """
        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)

        obs = torch.tensor(obs, dtype=torch.float32)
        act = torch.tensor(act, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        loss = self.alg.learn(obs, act, reward)
        # print(f"loss is {loss.detach().numpy()}")
        return loss.detach().numpy()

    def save(self,save_path=None,i=0):
        if not os.path.exists('../module'):
            os.makedirs('../module')
        torch.save(self.alg.model.state_dict(),f"../module/PG_{i}.pth")
        print("=====模型已保存====")
