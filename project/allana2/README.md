# Title
The project involves modifying the "intrinsic reward" term in VIME: Variational Information Maximizing Exploration, from 2016. In the paper, intrinsic reward is defined as

![equation](https://latex.codecogs.com/gif.latex?r'(s_t,a_t,s_{t&plus;1})=r(s_t,a_t,s_{t&plus;1})&plus;\eta&space;D_{KL}\left[p\left(\theta|\xi_t,a_t,s_{t&plus;1}&space;\right&space;)||p\left(\theta|\xi_t&space;\right&space;)&space;\right&space;])

where ![equation](https://latex.codecogs.com/gif.latex?p(\theta)) is the prior model on a randomly-valued dynamical models and
![equation](https://latex.codecogs.com/gif.latex?p\left(\theta|\xi_t,a_t,s_{t&plus;1}&space;\right&space;))
is the predicted posterior distribution as a result of variational inference.

![equation](https://latex.codecogs.com/gif.latex?p\left(\theta|\xi_t&space;\right&space;))
