import numpy as np

class Policy:
    def __init__(self, state_space=25, action_space=4):
        self.theta = np.zeros((state_space, action_space))
        self.action_space = action_space
        self.state_space = state_space
    
    def get(self, state):
        weights = self.theta[state]
        probs = np.exp(weights-np.max(weights)) / np.sum(np.exp(weights - np.max(weights)), axis=0) #softmax
        action = np.random.choice(np.arange(self.action_space), p=probs) #sample action
        return action, probs
        
    def update(self, update):
        self.theta += delta
    
    def gradient(self, probs, action):
        p_action = probs[action]
        grad = np.full(self.action_space, -p_action)
        grad[action] = 1-p_action
        return grad