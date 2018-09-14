import gym
import AcroBotLib
env = gym.make('AcroBotvMassi-v0')
for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        env.render('human')
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if reward:
            print("Current reward =  {} ...... ".format(reward))
        else:
            print("Current reward =  {}$ ".format(reward))
           
