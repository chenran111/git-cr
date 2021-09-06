import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random
import xlrd
from datetime import datetime
import os.path
import os
from openpyxl.reader.excel import load_workbook
from tkinter import _flatten

class Environ():
    

    def __init__(self):
        
        #设置参数
        self.C = 6     #边缘的最大存储容量
        self.U = 5     #边缘覆盖范围内的用户数量
        self.N = 10    #视频的数量
        self.l = 1.0   #本地到服务器的单位延迟
        self.p = 1.0   #本地到服务器的单位流量成本
        
        self.alpha = 0.8  #对延迟的关注度
        
        
        self.seed()
        self.state = None
        
        self.min_action = np.zeros(self.N)
        self.max_action = np.ones(self.N)
        
        self.steps = 0
        self._max_episode_steps = 12  #暂设
        self.steps_beyond_done = None
        
        self.min_observation = [-1 for _ in range(self.C + self._max_episode_steps * self.U)]
        self.min_observation = np.array(self.min_observation)
        self.min_observation = np.append(self.min_observation, 0)
        self.max_observation = [self.N-1 for _ in range(self.C + self._max_episode_steps * self.U)]
        self.max_observation = np.array(self.max_observation)
        self.max_observation = np.append(self.max_observation, 11)
        
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action
        )
        self.observation_space = spaces.Box(
            low=self.min_observation,
            high=self.max_observation
        )  

        #choice中用类似队列的形式来记录已存储的视频下标
        self.start = 0
        self.length = 0
        self.choice = [-1 for _ in range(self.C)]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    def step(self, action):
        
        #判断动作是否符合规范
        assert np.array(action) in self.action_space,"%r (%s) invalid" % (action, type(action))
        sum = 0
        for i in range(len(action)):
            sum = sum + action[i]
        assert sum == 1 , "%r (%s) invalid" % (action, "每次只缓存一个视频")
        
        
        uservideo = [[3,3,3,5,5,5,6,6,2,2,9,9],
                 [2,2,0,0,0,4,4,1,1,8,8,8],
                 [7,7,7,4,4,3,3,6,6,6,2,2],
                 [4,4,4,2,2,2,8,8,8,5,5,5],
                 [3,3,3,2,2,6,6,4,4,0,0,0]]
        uservideo = np.array(uservideo)
        uservideoOne = uservideo.flatten()  
        
        
        done = self.steps >= self._max_episode_steps
        done = bool(done)
        ind = -1  #做的动作，缓存的那个视频，得到其下标
        for i in range(self.N):
            if action[i] == 1:
                ind = i
        
        if not done:
            #edge缓存的视频没有被缓存过
            if ind not in self.choice:
                if self.length < self.C:
                    self.length += 1
                elif self.length == self.C:
                    self.start = (self.start + 1) % self.C
                else:
                    raise RuntimeError()
                self.choice[(self.start + self.length - 1) % self.C] = ind
                    
            
            #5个用户的奖赏
            userReward = [0]*self.U
            for i in range(self.U):
                for j in range(self.N):
                    if j in uservideo[i]:
                        if j in self.choice:  #如果用户观看的视频在边缘里的话，奖赏更高，给一个正值奖赏
                            userReward[i] += 5
                        else:  #如果用户观看的视频不在边缘里的话，延迟更高，奖赏更低
                            userReward[i] += -(self.alpha * self.l + (1-self.alpha) * self.p)
                        
            reward = 0.0
            print("self.steps,uservideo,action,self.choice,userReward:",self.steps,uservideo,action,self.choice,userReward)
            for i in range(self.U):
                reward = reward + userReward[i]
            self.steps += 1
            
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = -(self.alpha * self.l + (1-self.alpha) * self.p)
        else:
            if self.steps_beyond_done == 0:
                logger.warn("""
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = -(self.alpha * self.l + (1-self.alpha) * self.p)
            
        self.state = np.append(self.choice,uservideoOne)
        #original_time = datetime.now()
        a = random.randint(0, self._max_episode_steps-1)
        self.state = np.append(self.state,a)
        
        
        return np.array(self.state), reward, done, {}
        

    def reset(self):
        #边缘中已经缓存了哪些视频
        self.Choice = [-1 for _ in range(self.C)]
        resetChoiceNum = np.random.randint(self.C+1)    #边缘中已缓存视频的数量
        resetChoiceIndex = random.sample(range(0,self.N),resetChoiceNum)      #边缘中已缓存视频的下标
        for i in range(resetChoiceNum):
            self.Choice[i] = resetChoiceIndex[i]
        
        
        #用户观看的视频情况
        resetUservideo = [[random.randint(-1, self.N-1) for j in range(self._max_episode_steps)] for i in range(self.U)]
        resetUservideo = np.array(resetUservideo)
        resetUservideoOne = resetUservideo.flatten()
        
        self.state = np.append(self.Choice,resetUservideoOne)
        #original_time = datetime.now()
        a = random.randint(0, self._max_episode_steps-1)
        self.state = np.append(self.state,a)

        return np.array(self.state)
