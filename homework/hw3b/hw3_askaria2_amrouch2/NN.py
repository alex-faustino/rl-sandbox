import tensorflow as tf
import numpy as np


class FFNeuralNet(object):


    def __init__(self,input_size,layer1,output_size):
        #Define a FeedFroward network that take state as input and outputs the the Qfunction that correspend to each action i.e Q(a_i) = o_i, for i=1,2,3

        self.input_observation = tf.placeholder(shape= [None,input_size] , dtype= tf.float32)
        self.input_action = input_action = tf.placeholder(shape= [None,output_size] , dtype= tf.float32)
        
        # now declare the weights connecting the input to the hidden layer
        self.W1 = tf.Variable(tf.random_normal([input_size, layer1], stddev=0.03), name='W1')
        self.b1 = tf.Variable(tf.random_normal([layer1]), name='b1')
        
        # and the weights connecting the hidden layer to the output layer
        self.W2 = tf.Variable(tf.random_normal([layer1, output_size], stddev=0.03), name='W2')
        self.b2 = tf.Variable(tf.random_normal([output_size]), name='b2')

        # calculate the output of the hidden layer
        self.hidden_out = tf.nn.relu(tf.add(tf.matmul(self.input_observation,self.W1), self.b1))

        # now calculate the hidden layer output - in this case, let's use a softmax activated
        # output layer
        self.predict = tf.nn.softmax(tf.add(tf.matmul(self.hidden_out,self.W2), self.b2))
        #returns the Q value that correspend to the the index(input_action) != 0
        self.Q = tf.reduce_sum (tf.multiply(self.input_action,self.predict))

        
    def get_Qvalue(self):
            #return pointers to the input_actions and Q value
        return self.Q, self.input_action,self.input_observation

    def get_Output_NN(self):
        return self.predict
    def get_Input(self):
        return self.input_observation
        


class DQNet:

    def __init__(self):

        self.num_of_actions = output_size = 3
        self.num_of_states = input_size = 6
        self.epsilon = 0.1
        self.layer1 = layer1 = 30

        self.learningRate = 1e-3
        self.trainer = tf.train.AdamOptimizer(self.learningRate)

        self.input_size = self.num_of_states 
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

        
        self.y_target = y_target = tf.placeholder(shape= [None,1] , dtype= tf.float32)
        Qvalue,_,_ = Q.get_Qvalue()
        
        self.loss = loss = tf.reduce_sum(tf.square(y_target-Qvalue))
        self.optim = self.trainer.minimize(loss)
        
        self.maxActionQ = tf.argmax(Q.get_Output_NN(),axis = 0)
        self.maxQ_pred = tf.reduce_max(Q_pred.get_Output_NN(),axis = 0)
        sess.run(tf.global_variables_initializer())
        
    def updateQ(self,feed_dico):
        _,actions,states = self.Q.get_Qvalue()
        y = self.y_target
        feed_ = {states: feed_dico['s'],actions: feed_dico['a'],y: feed_dico['y']}
        #feed_ = {states: np.array(feed_dico['s']).T,
        #         actions: np.array(feed_dico['a']).T,
        #         y: np.array(feed_dico['y']).T}

        self.sess.run(self.optim,feed_dict=feed_)


    def MakeQpredEqQ(self):
        initQp = self.initQp
        initQ = self.initQ
        
        self.sess.run([initQp[i].assign(initQ[i]) for i in range(len(initQ))])
        return 

        
    def Session(self):
        return self.sess

    def MaxActionQ_pred(self,s):
         Action = self.sess.run(self.maxQ_pred,feed_dict= {self.Q_pred.get_Input():np.reshape(s,[1,self.num_of_states])})[0]
         return Action
         
    def NextAction(self, s,random_sample_action_Handler, greedy = True):

        # implements a greedy method for selection of action
        #greedy = 0 for greedy 1 otherewise
        
        if greedy:
            greed_sel = np.random.random_sample()
        else:
            greed_sel = 1

        #predict = self.sess.run(self.Q.get_Output_NN(),)

        predicted_Action = None
        
        if greed_sel >= self.epsilon:
            predicted_Action = self.sess.run(self.maxActionQ,feed_dict= {self.Q.get_Input():np.reshape(s,[1,self.num_of_states])})[0]
            
        else:
            predicted_Action =  random_sample_action_Handler()

        return predicted_Action
        



