from gym.envs.registration import register

register(
    id='AcroBotvMassi-v0',
    entry_point='AcroBotLib.AcroBot_vMassi:AcroBotMEnv',
)
