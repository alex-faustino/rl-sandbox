# AE598-RL Project
I plan to extend the part of results presented in ICRA-2018. In previous paper we demonstrated the policy transfer between two different domain tasks like mountain-car to Inverted-Pendulum and Cart-Pole to Bicycle. Also the policies transfered was for discrete action space, where the source policy was generated using DQN. For the class project I would be extending the proposed  Adaptive Transfer of Reinforcement learning policies for continuous action spaces between similar tasks with model uncertainties.

Experiments: Hopper-v1/HalfCheetah-v1/Ant-v1 ( using OpenAI RoboSchool )
Source Policy: DDPG/PPO


# Installation

perform the minimal installation of gym with 

git clone https://github.com/compdyn/598rl/tree/master/project/girishj2/project_girishj2.git

cd project_girishj2/gym-CartpoleEnv

pip install -e.

Similarly also install pendulum env

# Files

## Pendulum Results

1) Adaptive_Policy_transfer_v1.py : Adaptive Meta learning policy transfer code for Pendulum domain

2) ppo_standalone.py: PPO code for source ans target domain

## CartPole Results

policytransfer_transfer.py: Adaptive Meta learning policy transfer + Policy Learning code for Cartpole domain



