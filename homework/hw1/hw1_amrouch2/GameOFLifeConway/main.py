import gym
import GameOFLifeLib
env = gym.make('GameOfLifeEnv-v0')
for i_episode in range(1):
    action = env.action_space.sample()
    observation = env.reset(action)
    env.render('human')
    for t in range(10):
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render('human')
        if done:
            print("Episode finished after {} timesteps::: All the cells died :(".format(t+1))
            break
        
