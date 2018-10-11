AE 598RL project directory for Alex Faustino

### Option 1: Shared Autonomy
Shared autonomy was initially formulated as a POMDP by Javdani et al. where an agent must know the system dynamics, the goal space, and a user's policy ex-ante. The problem has been generalized to remove these constraints by Reddy et al.. In their formulation, the dynamics are estimated by DQN, the goal is predicted by classification or regression, and the user policy is unknown. The agent is rewarded in two parts: a parameterized reward for not catastrophically failing that is known to the agent and a reward given by the user at the end of each episode whether the goal is acheived or not. 

### Option 2: Adaptive Learning Rates

### Option 3: Reward Annealing
