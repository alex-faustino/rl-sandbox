from acrobot import AcroBotMEnv as AcrobotEnv
from time import sleep
from NN import DQNet
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Memory_Replay_Wrapper(object):
    def __init__(self,max_len=1e4):
        self.scenes = []
        self.max_len = max_len


    def push_scene(self,s_0,a_0,r_0,s_1,episode):
        wrap = {'s0':s_0,
                'a0':a_0,
                'r0':r_0,
                's1':s_1,
                'episode':episode}

        self.scenes.append(wrap)
        if len(self.scenes)> self.max_len:
            rem(self.scenes[0])

    def pop_scene(self,idx):
        if idx>self.max_len or idx<0:
            raise ValueError('Wrong Index in scenes')

        scene = self.scenes[idx]

        s0 = scene['s0']
        a0 = scene['a0']    
        r0 = scene['r0']
        s1 = scene['s1']
        episode_id = scene['episode']

        return s0,a0,r0,s1,episode_id

    def Sample(self):
        return self.pop_scene(np.random.uniform(0,len(self.scenes),1))


    def size(self):
        return len(self.scenes)



    
    if __name__ == '__main__':

        env = AcrobotEnv()
        Qlearn = DQNet()
        repMem = Memory_Replay_Wrapper(10000)
        episode_num = 3
        time_horizon = 1000
        size_batch = 30
        gamma = .95

        sess = Qlearn.Session()
        random_sample_action_Handler = env.action_space.sample
        for i_episode in range(episode_num):
            observ = env.reset()
            
            for t in range(time_horizon):
                env.render()
                
                # using greedy algorithm to select action
                new_action = Qlearn.NextAction(observ, random_sample_action_Handler, greedy =True )
                
                new_obser, reward = env.step(new_action)
                
                repMem.push_scene(observ,new_action,reward,new_obs,i_episode)
                
                y_target = []
                batch_s = []
                batch_a = []
                batch_y = []
                if repMem.size()<size_batch:
                    continue

                for i in range(size_batch):
                    s,a,r,s1,episode_id = repMem.Sample()
                    
                    if episode_id<episode_num:
                        y_target.append(r + gamma*Qlearn.MaxActionQ_pred(s1))
                    else:
                        y_target.append(r)
                        
                        batch_s.append(np.reshape(s,[Qlearn.num_of_states]))
                        
                        a_vec = np.zeros([Qlearn.num_of_actions,1])
                        a_vec[a] = 1
                        batch_a.append(a_vec)
                        batch_y.append(y_target)
                        
                        
                        feed_dico = {'s':batch_s,
                                     'a':batch_a,
                                     'y':batch_y}
                        
                        Qlearn.updateQ(feed_dico)
                        
                        


                        
                
            env.close()

