import tensorflow as tf
import numpy as np

class DQNet:

    def __init__(self):

        self.action = 3
        self.state = 4
        self.epsilon = 0.1
        self.layer1 = 30
        self.trainer = tf.train.AdamOptimizer()

        self.input_observation = tf.placeholder(shape= (self.state, 1) , dtype= tf.float32)

        # now declare the weights connecting the input to the hidden layer
        self.W1 = tf.Variable(tf.random_normal([self.state, self.layer1], stddev=0.03), name='W1')
        self.b1 = tf.Variable(tf.random_normal([self.layer1, 1]), name='b1')
        # and the weights connecting the hidden layer to the output layer
        self.W2 = tf.Variable(tf.random_normal([self.layer1, self.action], stddev=0.03), name='W2')
        self.b2 = tf.Variable(tf.random_normal([self.action, 1]), name='b2')

        # calculate the output of the hidden layer
        self.hidden_out = tf.add(tf.matmul(tf.transpose(self.W1), self.input_observation), self.b1)
        self.hidden_out = tf.nn.relu(self.hidden_out)

        # now calculate the hidden layer output - in this case, let's use a softmax activated
        # output layer
        self.predict = tf.nn.softmax(tf.add(tf.matmul(tf.transpose(self.W2), self.hidden_out), self.b2))

        self.init = tf.global_variables_initializer()
        self.init_l = tf.local_variables_initializer()
        tf.Session().run(self.init)
        tf.Session().run(self.init_l)

    def BackProp(self, Qnext):
        
        loss = tf.reduce_sum(tf.square(Qnext-self.predict))
        updateModel = self.trainer.minimize(loss)
        tf.Session().run(updateModel)

    def ActSel(self, s, greed):

        # implements a greedy method for selection of action
        
        if greed >= 0.5:
            greed_sel = np.random.random_sample()
        else:
            greed_sel = 1

        predict = tf.Session().run(self.predict, feed_dict={self.input_observation: s})


        if greed_sel >= self.epsilon:
            predict_max = tf.argmax(predict,0)
        else:
            predict_max = np.random.randint(low = 0, high = self.action)

        return predict_max, predict
        



