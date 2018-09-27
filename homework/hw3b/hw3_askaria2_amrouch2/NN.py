import tensorflow as tf
import numpy as np


class FFNeuralNet(object):


    def __init__(self,input_size,layer1,output_size):
        #Define a FeedFroward network that take state as input and outputs the the Qfunction that correspend to each action i.e Q(a_i) = o_i, for i=1,2,3

        self.input_observation = tf.placeholder(shape= [input_size, 1] , dtype= tf.float32)
        self.input_action = input_action = tf.placeholder(shape= [output_size, 1] , dtype= tf.float32)
        
        # now declare the weights connecting the input to the hidden layer
        self.W1 = tf.Variable(tf.random_normal([input_size, layer1], stddev=0.03), name='W1')
        self.b1 = tf.Variable(tf.random_normal([layer1, 1]), name='b1')
        
        # and the weights connecting the hidden layer to the output layer
        self.W2 = tf.Variable(tf.random_normal([layer1, output_size], stddev=0.03), name='W2')
        self.b2 = tf.Variable(tf.random_normal([output_size, 1]), name='b2')

        # calculate the output of the hidden layer
        self.hidden_out = tf.nn.relu(tf.add(tf.matmul(tf.transpose(self.W1), self.input_observation), self.b1))

        # now calculate the hidden layer output - in this case, let's use a softmax activated
        # output layer
        self.predict = tf.nn.softmax(tf.add(tf.matmul(tf.transpose(self.W2), self.hidden_out), self.b2))
        #returns the Q value that correspend to the the index(input_action) != 0
        self.Q = tf.reduce_sum (tf.multiply(self.input_action,self.predict))

        
    def get_Qvalue(self):
            #return pointers to the input_actions and Q value
        return self.Q, self.input_action 

    def get_Output_NN(self):
        return self.predict
    def get_Input(self):
        return self.input_observation
        


class DQNet:

    def __init__(self):

        self.action = output_size = 3
        self.state = input_size = 6
        self.epsilon = 0.1
        self.layer1 = layer1 = 30

        self.trainer = tf.train.AdamOptimizer()

        self.input_size = self.state 
        self.sess = sess =  tf.Session()

        #Action-Value Qfunc
        self.Q = Q = FFNeuralNet(input_size,layer1,output_size)
        self.initQ =  initQ = tf.trainable_variables()

        #Target Action-Value Qfunc
        self.Q_pred = Q_pred = FFNeuralNet(input_size,layer1,output_size)
        
        self.initQp = initQp = tf.trainable_variables()[len(initQ):]

        sess.run(tf.global_variables_initializer())
        #Initialize Qp = Q
        sess.run([initQp[i].assign(initQ[i]) for i in range(len(initQ))]) 

        
        self.y_target = y_target = tf.placeholder(shape= [1] , dtype= tf.float32)
        Qvalue,_ = Q.get_Qvalue()
        
        self.loss = loss = tf.reduce_sum(tf.square(y_target-Qvalue))
        self.optim = self.trainer.minimize(loss)
        self.maxActionQ = tf.argmax(Q.get_Output_NN(),axis = 0)

        
    def updateModel(self):
        return self.optim


    def Session(self):
        return self.sess

    def ActionSelection(self, s,random_sample_action, greedy = 1):

        # implements a greedy method for selection of action
        #greedy = 0 for greedy 1 otherewise
        
        if greedy >= 0.5:
            greed_sel = np.random.random_sample()
        else:
            greed_sel = 1

        #predict = self.sess.run(self.Q.get_Output_NN(),)


        if greed_sel >= self.epsilon:
            predicted_Action = self.sess.run(self.maxActionQ,feed_dict= {self.Q.get_Input():np.reshape(s,[self.state,1])})[0]
            
        else:
            predicted_Actionx =  random_sample_action()

        return predicted_Action
        



