= []
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
    plt.sh