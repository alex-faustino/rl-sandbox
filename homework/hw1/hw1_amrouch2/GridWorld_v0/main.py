import gym
import GridWorldLib
env = gym.make('WorldGrid-v0')
for i_episode in range(1):
    observation = env.reset()
    for t in range(10):
        env.render('human')
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
