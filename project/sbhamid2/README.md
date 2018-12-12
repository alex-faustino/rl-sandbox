# Urban Path Planning to Minimize GPS Integrity Risk Using Proximal Policy Optimization 

## Motivation and Objectives
GPS plays a pivotal role in the navigation of autonomous vehicles such as self-driving cars, unmanned aerial vehicles, and so on. However, navigation via GPS in urban areas suffer from multipath, satellite blockage and signal attenuation due to the presence of tall buildings. Given the grid layout of the urban areas, multiple routes are feasible to reach from a start point to an end point. Therefore, the objective of this work is to perform robust path planning to minimize the GPS integrity risk, while navigating in urban areas.

<img src="https://github.com/compdyn/598rl/blob/master/project/sbhamid2/Images/Motivation.png" width="45%"></img>

## Assumptions: 
* GPS satellites are considered static and GPS receiver clock bias errors are ignored
* Uniformly distributed buildings are considered, with heights ranging from 20-100m
* Key decision making points are at the intersections, and along the rest of the block, the same action is continued till the next intersection point is reached
* Discrete control actions are chosen-forward, backward, left or right

## Algorithm Details

### Environment 
* Multipath effects are induced due to the surrounding 4 buildings in all directions as seen in figure below
* 3D building models/LiDAR are used to obtain 3D point cloud data. The urban area considered is 500x500x100m area
* Constant goal is to reach (800,800,4)m coordinates. 
<img src="https://github.com/compdyn/598rl/blob/master/project/sbhamid2/Images/Env.JPG" width="45%"></img>
 
### Rewards 
Reward at each step is estimated as a summation of three components
* GPS Integrity risk calculated via GPS RAIM-based solution separation~[1]
* Difference in distances between the previous state to the goal and the current state to the goal
* On reaching the goal

### Architecture
Deep reinforcement learning based Proximal Policy Optimization algorithm is used to train the system. Actor and Critic are considered as separate networks. Actor has 3 hidden layers with 20 nodes each and critic has 4 hidden layers with 20 nodes each. PyTORCH is used as the neural network platform. EKF is used to obtain the predicted position at each instant based on which the observation vector is designed.  
<img src="https://github.com/compdyn/598rl/blob/master/project/sbhamid2/Images/Architecture.png" width="45%"></img>

### Observation vector
<img src="https://github.com/compdyn/598rl/blob/master/project/sbhamid2/Images/States.JPG" width="45%"></img>

## Results 
* Case 1 and 2: Open-sky and urban area-based grid world environment with 3D robot position as input. Comparison of PPO with Q-learning and SARSA
* Case 3 and 4: Open-sky and urban area with observations from GPS and 3D building models as input. Figure below shows that the PPO estimated policy shows lower positioning error, as indicated in blue, as compared to the shortest path to the goal, as shown in red. 
<img src="https://github.com/compdyn/598rl/blob/master/project/sbhamid2/Images/Result.png" width="45%"></img>

## References (Uploaded in this folder): 
[1] Joerger M, Chan FC, Pervan B. Solution Separation Versus Residual‐Based RAIM. Navigation: Journal of The Institute of Navigation. 2014 Dec;61(4):273-91.
[2] Tai L, Paolo G, Liu M. Virtual-to-real deep reinforcement learning: Continuous control of mobile robots for mapless navigation. InIntelligent Robots and Systems (IROS), 2017 IEEE/RSJ International Conference on 2017 Sep 24 (pp. 31-36). IEEE.
[3] Everett M, Chen YF, How JP. Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning. arXiv preprint arXiv:1805.01956. 2018 May 4.

