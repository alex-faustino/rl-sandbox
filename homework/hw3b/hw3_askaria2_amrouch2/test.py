from acrobot import AcrobotEnv
from time import sleep
#from NN import DQNet
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

action = 3
state = 4
epsilon = 0.1
layer1 = 30


input_observation = tf.placeholder(shape= (state, 1) , dtype= tf.float32)
Qnext = tf.placeholder(shape= (action,1) , dtype= tf.float32)

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([state, layer1], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([layer1, 1]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([layer1, action], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([action, 1]), name='b2')

# calculate the output of the hidden layer
hidden_out1 = tf.add(tf.matmul(tf.transpose(W1), input_observation), b1)
hidden_out = tf.nn.relu(hidden_out1)

# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
predict = tf.nn.softmax(tf.add(tf.matmul(tf.transpose(W2), hidden_out), b2))


        
loss = tf.reduce_sum(tf.square(Qnext-predict))
trainer = tf.train.AdamOptimizer()
updateModel = trainer.minimize(loss)

#init = tf.global_variables_initializer()

env = AcrobotEnv()
init = tf.global_variables_initializer()
episode_num = 1
time_horizon = 1000

#THE ISSUE WAS IN SESSION SO AT EACH INSTANCIATION THEY CREAT A NEW ENV THUS TWO DIFF CALL OF SESSION WILL LEAD TO TWO DIFF ENV 
#WHERE ONE HAS THE VAR INITIALIZED WHILE THE OTHER NOPE :P


for i_episode in range(episode_num):
    
    observation = env.reset()
    with tf.Session() as sess:
        for t in range(time_horizon):
            env.render()        
            sess.run(init)   
            # using greedy algorithm to select action
            greed_sel = np.random.random_sample()
            predict = sess.run(predict, feed_dict={input_observation: observation})
            if greed_sel >= epsilon:
                predict_max = tf.argmax(predict,0)
            else:
                predict_max = np.random.randint(low = 0, high = action)

            action = predict_max
            target_Q = predict

            observation, reward = env.step(action)

            action_old = action
            
            # action selection without greedy
            predict = sess.run(predict, feed_dict={input_observation: observation})
            action = tf.argmax(predict,0) 
            Q_val = predict

            target_Q[action_old] = reward + 0.95*np.max(Q_val)
            
            sess.run([updateModel, loss], feed_dict={input_observation: observation, Qnext: target_Q})
        
    env.close()

