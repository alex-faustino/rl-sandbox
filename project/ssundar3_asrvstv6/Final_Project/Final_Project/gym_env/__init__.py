from gym.envs.registration import register

register(
        id = 'grid_world-v0',
        entry_point = 'gym_env.Grid_World:Grid',
)

register(
        id = 'single_chain-v0',
        entry_point = 'gym_env.single_chainEnv:NChainEnv',
)

register(
        id = 'two_state-v0',
        entry_point = 'gym_env.two_state:two_stateEnv',
)

register(
        id = 'double_chain-v0',
        entry_point = 'gym_env.double_chain:double_chainEnv'
)
