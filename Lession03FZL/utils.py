import numpy as np

class replay_buffer():
    def __init__(self, mem_size, input_shape, n_actions):
        self.dir = "Replay buffer/"
        self.mem_cntr = 0
        self.mem_size = mem_size
        self.state_mem = np.zeros((mem_size,input_shape),
                                  dtype=np.float32)
        self.new_state_mem = np.zeros((mem_size,input_shape),
                                      dtype=np.float32)
        # self.action_mem = np.zeros((mem_size,n_actions),
        #                            dtype=np.int64)
        self.action_mem = np.zeros(mem_size,
                                   dtype=np.int64)
        self.reward_mem = np.zeros(self.mem_size,dtype=np.float32)
        self.done_mem = np.zeros(self.mem_size,dtype=np.bool8)
        
    def sample_memory(self,batch_size):
        max_mem = min(self.mem_cntr,self.mem_size)
        batch = np.random.choice(max_mem,batch_size,replace=False)
        state = self.state_mem[batch]
        action = self.action_mem[batch]
        reward = self.reward_mem[batch]
        state_ = self.new_state_mem[batch]
        done = self.done_mem[batch]
        return state, action, reward, state_, done
    
    def store_transition(self, state, action, reward, state_, done):
        i = self.mem_cntr % self.mem_size
        self.state_mem[i] = state
        self.action_mem[i] = action
        self.reward_mem[i] = reward
        self.new_state_mem[i] = state_
        self.done_mem[i] = done
        self.mem_cntr += 1
    
    def get_sel_transition(self, index):
        max_mem = min(self.mem_cntr,self.mem_size)
        idx = index if index < max_mem else 0
        state = self.state_mem[idx]
        action = self.action_mem[idx]
        reward = self.reward_mem[idx]
        state_ = self.new_state_mem[idx]
        done = self.done_mem[idx]
        return state, action, reward, state_, done