
import gym, random
import gym.spaces as spaces

class GridWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, hard_version=False):
        self.hard_version = hard_version
        self.last_action = None   # used for rendering
        self.observation_space = spaces.Discrete(25)
        self.action_space = spaces.Discrete(4)
        self.reset()

    def step(self, a):
        self.last_action = a
        done = False
        r = 0
        if self.s == 1: # first teleporting state
            self.s = 21
            r = 10
            return (self.s, r, done)
        elif self.s == 3: # second teleporting state
            self.s = 13
            r = 5
            return (self.s, r, done)
        else:
            # convert state to i,j coordinates
            i = self.s // 5
            j = self.s % 5
            if a == 0: # right
                j += 1
            elif a == 1: # up
                i -= 1
            elif a == 2: # left
                j -= 1
            elif a == 3: # down
                i += 1
            else:
                raise Exception(f'invalid action: {a}')
            if i < 0 or i >= 5 or j < 0 or j >= 5:
                # try to leave the domain, don't change self.s
                r = -1
            else:
                # convert the updated coordinates back to a state number
                self.s = i * 5 + j
        return (self.s, r, done)

    def reset(self):
        self.s = random.randrange(self.observation_space.n)
        self.last_action = None
        return self.s

    def render(self):
        k = 0
        output = ''
        for i in range(5):
            for j in range(5):
                if k == self.s:
                    output += 'X'
                elif k == 1 or k == 3:
                    output += 'o'
                else:
                    output += '.'
                k += 1
            output += '\n'
        if self.last_action is not None:
            print(['right', 'up', 'left', 'down'][self.last_action])
            print()
        print(output)
