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

### Environment and Rewards: 
* Multipath effects are induced due to the surrounding 4 buildings in all directions as seen in figure below
* 3D building models/LiDAR are used to obtain 3D point cloud data. The urban area considered is 700x700x150m area
* Constant goal is to reach (800,800,4)m coordinates. 
<img src="https://github.com/compdyn/598rl/blob/master/project/sbhamid2/Images/Env.png" width="45%"></img>
 
### Observation vector
<img src="https://github.com/compdyn/598rl/blob/master/project/sbhamid2/Images/States.JPG" width="45%"></img>

### Architecture
Deep reinforcement learning based Proximal Policy Optimization algorithm is used to train the 
<img src="https://github.com/compdyn/598rl/blob/master/project/sbhamid2/Images/Architecture.png" width="45%"></img>

## Results 
* Case 1 and 2: Open-sky and urban area-based grid world environment with 3D robot position as input. Comparison of PPO with Q-learning and SARSA
* Case 3 and 4: Open-sky and urban area with observations from GPS and 3D building models as input 
<img src="https://github.com/compdyn/598rl/blob/master/project/sbhamid2/Images/Result.png" width="45%"></img>

