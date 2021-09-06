
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *

# from ipdb import set_trace as debug

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, nb_states, nb_actions, args):
        
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)  #单个星号代表这个位置接收任意多个非关键字参数，在这个位置上将其转化成元组，而双星号代表这个位置接收任意多个关键字参数，在该位置上将其转化成字典  
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)   #可用于迭代优化的参数或者定义参数组的dicts   lr：学习率，默认1e-3，更新梯度的时候使用    args.prate是什么？？

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)   #args.rate是什么？？

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight  把self.actor中的参数复制到self.actor_target中
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer     缓冲区没看懂
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters    args中的参数？？
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True
        
        self.value_loss = None
        self.policy_loss = None
        self.reward_batch = None
        self.q_batch = None
        self.target_q_batch = None
        

        # 什么意思？？？？
        if USE_CUDA: self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
        
        self.reward_batch = reward_batch

        # Prepare for the target q batch
        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(to_tensor(next_state_batch, volatile=True)),
        ])
        next_q_values.volatile=False

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        
        self.target_q_batch = target_q_batch
        
        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
        
        self.q_batch = q_batch

        
        self.value_loss = criterion(q_batch, target_q_batch)
        self.value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        self.policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        self.policy_loss = self.policy_loss.mean()
        self.policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        
    def get_value_loss(self):
        return self.value_loss
    def get_policy_loss(self):
        return self.policy_loss
    def get_reward_batch(self):
        return self.reward_batch
    def get_q_batch(self):
        return self.q_batch
    def get_target_q_batch(self):
        return self.target_q_batch

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        
        #action = np.random.uniform(-1.,1.,self.nb_actions)
        index = np.random.randint(0,self.nb_actions)
        action = [0] * self.nb_actions
        action[index] = 1
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)   #squeeze(0)代表若第一维度值为1则去除第一维度
        action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        
        maxofaction = 0
        index = -1
        
        for i in range(len(action)):
            if action[i]>maxofaction:
                maxofaction = action[i]
                index = i
        action = [0]*10
        action[index]=1
        #action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
