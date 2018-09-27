from DQNtest import DQNAgent as Agent
from acrobot_threeactions import AcroBot
env = AcroBot(2)
agent = Agent(env)
agent.train()
