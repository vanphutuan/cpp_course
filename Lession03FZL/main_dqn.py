from deep_q_network import DQNagent
import numpy as np
import gym
from gym.spaces import Discrete
import matplotlib.pyplot as plt

def vpt(i):
    print(f"##########{i}##########")

def main():
    n_games = 100
    buffer_size = 50000
    batch_size = 32
    lr = 1e-4
    gamma = 0.99
    eps_max = 1
    eps_min = 0.01
    eps_dec = 1e-5

    # env = gym.make('FrozenLake-v1',is_slippery=False)
    env = gym.make('FrozenLake-v1')

    if isinstance(env.observation_space,Discrete):
        n_obs = 1
    if isinstance(env.action_space,Discrete):
        n_act = 1
        ndim_act = env.action_space.n 
    agent = DQNagent(n_obs,n_act,ndim_act,lr,
                 buffer_size,batch_size,
                 gamma,eps_max,eps_min,eps_dec)
    scores = []
    win_pct = []
    for i in range(n_games):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            obs_, reward, done, info, _ = env.step(act)
            agent.store_transistion(obs, act, reward, obs_, done)
            agent.learn()
            obs = obs_
            score += reward
        scores.append(score)
        if i%10 == 0:
            average = np.mean(scores[-20:])
            win_pct.append(average)
            if i%100 == 0:
                print(f"--epoch:{i}--score:{average}--",
                      f"eps:{agent.epsilon}")
    
    plt.plot(win_pct)
    plt.show()

if __name__ == "__main__":
    main()