from gym.envs.registration import register

register(
    id='TargetFinder-v0',
    entry_point='target_finder.target_finder:TargetFinder',
)
