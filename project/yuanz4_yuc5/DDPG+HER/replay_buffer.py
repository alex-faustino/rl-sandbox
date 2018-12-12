import numpy as np

class ReplayBuffer:
    def __init__(self, limit=max):
        self.Buffer = []
        self.max = limit
        self.size = 0
    def store(self, experience):
        if self.size == self.max:
            self.Buffer[np.random.randint(self.max)] = experience
        else:
            self.Buffer.append(experience)
            self.size += 1
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        state_batch = np.array([])
        action_batch = np.array([])
        reward_batch = np.array([])
        next_state_batch = np.array([])
        terminal_batch = np.array([])
        for i in range(batch_size):
            if i == 0:
                state_batch = self.Buffer[idx[i]][0]
                action_batch = self.Buffer[idx[i]][1]
                next_state_batch = self.Buffer[idx[i]][3]
            else:
                state_batch = np.vstack([state_batch, self.Buffer[idx[i]][0]])
                action_batch = np.vstack([action_batch, self.Buffer[idx[i]][1]])
                next_state_batch = np.vstack([next_state_batch, self.Buffer[idx[i]][3]])
            reward_batch = np.append(reward_batch, self.Buffer[idx[i]][2])
            terminal_batch = np.append(terminal_batch, self.Buffer[idx[i]][4])
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch
