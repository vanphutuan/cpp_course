import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import replay_buffer

class DQN(nn.Module):
    def __init__(self, n_obs, ndim_act, gamma, lr):
        super(DQN,self).__init__()
        self.dir = "##### DQN/"
        self.n_obs = n_obs
        self.ndim_act = ndim_act
        self.gamma = gamma
        self.fc1 = nn.Linear(self.n_obs, 64)
        self.fc2 = nn.Linear(64, self.ndim_act)
        self.optim = optim.Adam(self.parameters(),lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        action = self.fc2(x)
        return action
        

class DQNagent:
    def __init__(self,n_obs,n_act,ndim_act,lr,
                 buffer_size,batch_size,
                 gamma,eps_max,eps_min,eps_dec):
        self.dir = "##### DQNagent/"
        self.n_obs = n_obs
        self.n_act = n_act
        self.ndim_act = ndim_act
        self.gamma = gamma
        self.epsilon = eps_max
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.learn_step_cnt = 0
        self.batch_size = batch_size
        self.update_target_cnt = 100
        self.action_space = [i for i in range(ndim_act)]
        self.memory = replay_buffer(buffer_size, n_obs, n_act)
        self.q_eval = DQN(n_obs,ndim_act,gamma,lr)
        self.q_target = DQN(n_obs,ndim_act,gamma,lr)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            # kiem tra int
            if isinstance(state,int):
                state = np.array([state])
            else:
                print("Need modification for the new state's dtype")
            state = T.tensor(state,dtype=T.float32).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        return action
    def sample_replaybuffer(self):
        states, actions, rewards, states_, dones = \
            self.memory.sample_memory(self.batch_size)
        states = T.tensor(states).to(self.q_eval.device)
        actions = T.tensor(actions).to(self.q_eval.device)
        rewards = T.tensor(rewards).to(self.q_eval.device)
        states_ = T.tensor(states_).to(self.q_target.device)
        dones = T.tensor(dones).to(self.q_eval.device)
        return states, actions, rewards, states_, dones
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        # print(self.dir,"learn")
        self.q_eval.optim.zero_grad()
        self.update_q_target()
        states, actions, rewards, states_, dones = \
            self.sample_replaybuffer()
        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)
        q_pred = q_pred[indices,actions]
        q_next = self.q_target(states_).max(dim=1)[0]
        # .max(dim=1) => (tensor(max_val,grad), tensor(indices))
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next
        loss = self.q_eval.loss(q_target,q_pred)
        loss.backward()
        self.q_eval.optim.step()
        self.eps_decrement()
        self.learn_step_cnt += 1
        
        

    def update_q_target(self):
        if self.learn_step_cnt % self.update_target_cnt == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

    def store_transistion(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, 
                                     reward, state_, done)

    def eps_decrement(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

