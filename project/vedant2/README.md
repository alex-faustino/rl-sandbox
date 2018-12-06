# General attitude control for satellite

"""
-*- coding: utf-8 -*-

Created on Thu Oct 11 10:41:07 2018

@author: Vedant
"""



- Builiding up on the HW1 enviroment, using the satellite attitude control environment, train RL agents developedin cals t solve a generic rigid body attitude control problem.

- This problem will be formulated in a similar way as the hand cube manuplation paper.

- The only initial constraint will be limits on control torques.

- System inertia and mass will change on larger scale for each mini batch to avoid overfitting, and small variations between each episodes.

- Once a generic attitude control agent is realized, implement realistic constraints and compare training times, learning rates for different agents (Agents learning from scratch vs agents learning from the agents trained in the generic attitude control problem).

- Reward shaping will be used from my Master's Thesis, which prouced trajectories using Dynamic Programming.

-Since Dynamics are similar for the Pendulum, algorithms deploywd on the simple pendulum will be the first candidates.

References

- M. Andrychowicz, B. Baker, M. Chociej, R. Jozefowicz, B. McGrew, J. Pachocki, A. Petron, M. Plappert, G. Powell, A. Ray, J. Schneider, S. Sidor, J. Tobin, P. Welinder, L. Weng, W. Zaremba (2018) "Learning Dexterous In-Hand Manipulation", arXiv:1808.00177.

- David Silver, Thomas Hubert,Julian Schrittwieser,Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot,Laurent Sifre, Dharshan Kumaran, Thore Graepel,Timothy Lillicrap, Karen Simonyan, Demis Hassabis (2017) "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm", arXiv:1712.01815

 
