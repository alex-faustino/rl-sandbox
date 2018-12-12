# Active Sensing for Robot Localization

Active sensing is the problem of intelligent control strategies applied to the data acquisition process. Controlling the gaze of field of view (FOV) sensors is important to enable localization.

## Project Goal

* Learn policy to control FOV sensors to explore and track landmarks (LMs) in the environment to enable localization
* Extend policy to control two FOV sensors

<img src="https://github.com/compdyn/598rl/blob/master/project/ashetty2/images/goal1.png" width="45%"></img>
<img src="https://github.com/compdyn/598rl/blob/master/project/ashetty2/images/goal2.png" width="45%"></img>

## Environment

### State Vector:
<img src="https://github.com/compdyn/598rl/blob/master/project/ashetty2/images/state.PNG" width="90%"></img>

### Reward:
The reward at each step is the number of landmarks within the sensor FOVs. For example, in the following case the reward would be 3:

<img src="https://github.com/compdyn/598rl/blob/master/project/ashetty2/images/reward.png" width="50%"></img>

### Network Architecture and Implementation:
* Single network for actor and value estimation
* 2 hidden layers with 100 nodes, with tanh non-linearities
* Continuous PPO with multi-processing on Google Cloud Platform

## Results

Single FOV achieved an average reward per step of ~1.9. For longer trajectories, the sensor needs to explore again.
<img src="https://github.com/compdyn/598rl/blob/master/project/ashetty2/images/1fovshort.PNG" width="45%"></img>
<img src="https://github.com/compdyn/598rl/blob/master/project/ashetty2/images/1fovlong.PNG" width="45%"></img>

Two FOV sensors achieved an average reward per step on ~1.4. Did not perform as well as expected and need to try out a few more variations to the network structure and the state vector format.
<img src="https://github.com/compdyn/598rl/blob/master/project/ashetty2/images/2fov.PNG" width="50%"></img>

## Directory Info

_FastSLAM_: Small environment with single FOV sensor

_FastSLAM1_: Longer environment with single FOV sensor

_FastSLAM2_: Longer environment with two FOV sensors

_train.py_: Performs the learning by using agent defined in *hw6_ppo.py*

_test.ipynb_: Can be used to visualize desired model

*saved_models*: Contains saved models. Use *case3_LM_best.pth* for trained single FOV cases. Use *case5_LM_best.pth* for trained two FOV cases

_evaluate.ipynb_: Evaluates models to obtain average reward per step metric

_workspace.ipynb_: Notebook to tinker around with the environments
