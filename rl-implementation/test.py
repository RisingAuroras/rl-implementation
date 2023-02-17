import gym,time
import random
import numpy as np
import torch
import torch.nn.functional as F
# from Agent import Agent
from dqn.agent import Agent
from dqn.algorithm import DQN
from dqn.model import Model

env = gym.make('CartPole-v0')
obs_dim = env.observation_space.shape[0]  # CartPole-v0: (4,)
act_dim = env.action_space.n  # CartPole-v0: 2
model = Model(obs_dim=obs_dim, act_dim=act_dim)
dict = torch.load('module/DQN_4.pth')
model.load_state_dict(dict)
algorithm = DQN(model, gamma=0., lr=0.)
agent = Agent(
    algorithm,
    act_dim=act_dim,
    e_greed=0.1,  # 有一定概率随机选取动作，探索
    e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低
for i_episode in range(1):
    observation = env.reset() #初始化环境每次迭代
    i = 0
    while True:
        i += 1
        env.render() #显示
        action = agent.predict(observation)
        o, reward, done, info = env.step(action)
        if done:
            break
        observation = o
    print(i)
env.close()