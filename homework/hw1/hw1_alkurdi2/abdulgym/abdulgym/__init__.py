

from gym.envs.registration import register, registry, make, spec



register(
    id='AAgridworld-v0',
    entry_point='abdulgym.envs:AAgridworldEnv',
)

register(
    id='AAacrobot-v0',
    entry_point='abdulgym.envs:AAacrobotEnv',
)
register(
    id='Weedbot-v0',
    entry_point='abdulgym.envs:WeedbotEnv',
)