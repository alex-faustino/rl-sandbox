from lib import *

class QlearningAlgo(object):
    def __init__(self,envObj,epsilon,gamma):
        self.env = envObj
        self.nA = nA = self.env.nA
        self.nS = nS = self.env.nS
        self.epsilon = epsilon
        self.Q = self.alpha = np.zeros((nS, nA))
        self.alpha += 1
        self.gamma = gamma
        self.isSave = False
        self.posX = []
        self.posY = []
        self.cummReward = []
        self.time = []
        
    def greedy(self,s):
        self.env.seed()
        if self.env.np_random >= self.epsilon:
            return np.argmax(self.Q[s][:])
        else:
            return env.action_space.sample()


    def updateSARSA(self, a,anext, s, snext, r):
        alpha = np.power(1./self.alpha[s][a],.8)
        self.Q[s][a] += alpha * (r + self.gamma*self.Q[snext][anext]-self.Q[s][a])
        self.alpha[s][a] += 1.


    def updateQ(self, a, s, snext, r):
        alpha = np.power(1./self.alpha[s][a],.8)
        self.Q[s][a] += alpha * (r + self.gamma*np.max(self.Q[snext][:])-self.Q[s][a])

        self.alpha[s][a] +=  1.

    def saveHistory(self,pos,r,t):
        if self.isSave:
            #row*ncol + col
            self.posX = np.append(self.posX,int(pos/self.env.ncol))
            self.posY = np.append(self.posY,pos%self.env.ncol)
            self.cummReward = np.append(self.cummReward,r)
            self.time = np.append(self.time,t)
    def activeSave(self):
        self.isSave = not self.isSave
        return self.isSave

    def getHistory(self):
        return (self.posX,self.posY,self.cummReward,self.time)
    
    def test(self):
        print self.nA
        print self.nS
        
        
