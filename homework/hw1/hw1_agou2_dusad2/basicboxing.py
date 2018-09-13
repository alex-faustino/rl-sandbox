from gym import spaces, core

class Agent():
    def __init__(position=0, action=0):
        self.position = position
        self.action = action   

class BoxingEnv(core.Env):

    def __init__(self, length=5, initial_health=5):
        self.length = length
        self.initial_health = initial_health
        self.state_space = spaces.Dict({
            "p1": spaces.Discrete(length),
            "p2": spaces.Discrete(length),
            "h1": spaces.Discrete(initial_health),
            "h2": spaces.Discrete(initial_health)
        })
        self.action_space = spaces.Discrete(5)
        self.action_map = {
            0: "noop",
            1: "left",
            2: "right",
            3: "punchleft", # left side, not left hand
            4: "punchright" # right side, not right hand
        }
        self.state = None
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = {}
        self.state['p1'] = self.state_space.spaces['p1'].sample()
        self.state['p2'] = self.state_space.spaces['p2'].sample()
        self.state['h1'] = self.initial_health
        self.state['h2'] = self.initial_health
        return self.state

    def step(self, action):
        a1, a2 = action
        p1 = self.state['p1']
        p2 = self.state['p2']
        h1 = self.state['h1']
        h2 = self.state['h2']
        r1, r2 = 0, 0
        terminal = False
        if self.action_map[a1] == "left":
            if p1 == 0:
                r1 -= 1
            else:
                p1 -= 1
        elif self.action_map[a1] == "right":
            if p1 == self.length-1:
                r1 -= 1
            else:
                p1 += 1
        elif self.action_map[a1] == "punchleft":
            if p1 - 1 == p2:
                r1 += 1
                if h2 == 0:
                    r2 -= 10
                    terminal = True
                h2 -= 1
                r2 -= 1
        elif self.action_map[a1] == "punchright":
            if p1 + 1 == p2:
                r1 += 1
                if h2 == 0:
                    r2 -= 10
                    terminal = True
                h2 -= 1
                r2 -= 1
        
        if self.action_map[a2] == "left":
            if p2 == 0:
                r2 -= 1
            else:
                p2 -= 1
        elif self.action_map[a2] == "right":
            if p2 == self.length-1:
                r2 -= 1
            else:
                p2 += 1
        elif self.action_map[a2] == "punchleft":
            if p2 - 1 == p2:
                r2 += 1
                if h2 == 0:
                    r1 -= 10
                    terminal = True
                h1 -= 1
                r1 -= 1
        elif self.action_map[a2] == "punchright":
            if p2 + 1 == p2:
                r2 += 1
                if h2 == 0:
                    r1 -= 10
                    terminal = True
                h1 -= 1
                r1 -= 1
        
        self.state = {
            'p1': p1,
            'p2': p2,
            'h1': h1,
            'h2': h2
        }
        return self.state, (r1, r2), terminal, None

    def render(self, mode='ansi'):
        p1 = "|"
        for i in range(self.length):
            if self.state['p1'] == i:
                p1.append(" " + ("" if self.state['h1'] >= 10 else " ") + self.state['h1'])
            else:
                p1.append(" __")
        p1.append(" |")
        p2 = "|"
        for i in range(self.length):
            if self.state['p2'] == i:
                p1.append(" " + ("" if self.state['h2'] >= 10 else " ") + self.state['h2'])
            else:
                p1.append(" __")
        p2.append(" |")
        return p1 + "\n" + p2 + "\n"

    def close(self):
        if self.viewer: self.viewer.close()

# env = BoxingEnv()
# env.reset()
# net_r1, net_r2 = 0, 0
# for i in range(1000):
#     obs, reward, terminal, _ = env.step((env.action_space.sample(), env.action_space.sample()))
#     r1, r2 = reward
#     net_r1 += r1
#     net_r2 += r2
#     print(i, obs, reward, net_r1, net_r2)
#     if terminal:
#         break
