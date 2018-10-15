# Concise Background Information
The project involves modifying the "intrinsic reward" term in VIME: Variational Information Maximizing Exploration, from 2016. In the paper, intrinsic reward is defined as

![equation](https://latex.codecogs.com/gif.latex?r'(s_t,a_t,s_{t&plus;1})=r(s_t,a_t,s_{t&plus;1})&plus;\eta&space;D_{KL}\left[p\left(\theta|\xi_t,a_t,s_{t&plus;1}&space;\right&space;)||p\left(\theta|\xi_t&space;\right&space;)&space;\right&space;])

where ![equation](https://latex.codecogs.com/gif.latex?\xi_t=\left\{s_1,a_1,\dots,s_t\right\}) is the history of the agent at time ![equation](https://latex.codecogs.com/gif.latex?t), ![equation](https://latex.codecogs.com/gif.latex?p\left(\theta|\xi_t&space;\right&space;)) is the prior model on a randomly-valued dynamical models and
![equation](https://latex.codecogs.com/gif.latex?p\left(\theta|\xi_t,a_t,s_{t&plus;1}&space;\right&space;))
is the predicted posterior distribution.

The actual method is by approximate variational inference, rather than a simple computation of probabilities and rewards. 

# Proposed Contribution
The term ![equation](https://latex.codecogs.com/gif.latex?\eta) is a hyperparameter of the VIME algorithm. My project will involve replacing intrinsic reward with the Entropic Value at Risk (EVaR). In using EVaR, we choose ![equation](https://latex.codecogs.com/gif.latex?\eta) as the optimization parameter. In-so-doing, we achieve an uppder bound to the original intrinsic reward expression. Moreover, we also gain positive homogeneity and tranlsation invariance with respect to the reward model. As a result, the expected performance of the algorithm is expected to be more robust to the designer changing the rewards in the environment by the same linear transformation.
