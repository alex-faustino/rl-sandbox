## Denis's project
# Multi-Agent RL (MARL) in Capture the Flag (CtF) environment.

### Overview
This project considers the problem of cooperative learners in partially observable, stochastic environment, receiving feedback in the form of joint reward. This work will use a flexible multi-agent competitive environment of my own design for online training and direct policy performance comparison called Capture the Flag (CtF) and available at https://github.com/osipychev/stf-public or at https://denisos.com .

This simulation forms a formal problem of a multi- agent Reinforcement Learning (RL) under partial observability, where the goal is to maximize the score performance measured in a direct confrontation. To address the complexity of the problem I propose a distributed deep stochastic policy gradient with individual observations, experience replay, policy transfer, and self-play.

### Motivation
Many real-life tasks involve partial observability and multi-agent planning. Traditional RL approaches such as Q-Learning and policy-based methods are poorly suited to multi-agent problems. Actions performed by third agents are usually observed as transition noise that makes the learning very unstable (Nowe ́ et al., 2012; Omidshafiei et al., 2017). Policy gradient methods, usually exhibit very high variance when coordination of multiple agents is required (Lowe et al., 2017). The complex interaction between the agents makes learning difficult due to the agent’s perception of the environment as non-stationary and partially observable. Nevertheless, multi-agent systems are finding applications in high demand problems including resource allocation, robotic swarms, dis- tributed control, collaborative decision making, real-time strategy (RTS) and real robots. But they are a substantially more complex task for online learning algorithms and often require multi-layer solutions.

### Environment Description
The CtF simulation is a gladiator pit for algorithms. This Python-based environment is designed to throw together various AI algorithms or a human. Two teams confront each other and the goal is to capture the other team’s flag or destroy the enemy units. Two types of units exist in each team and they have different abilities. The observation is limited by the fog of war shown on the left and provides the information available to the team. The true state of the environment shown on the right and reflects all the teams.

### Literature on Multiagent RL
[2] - Survey on multiagent systems focused on stability of the learning dynamics and adaptation to the changing behavior of the other agents. Considers fully cooperative, fully competitive, or mixed setting.

[12] - Multi-agent system (MAS) overview (from robotics perspective), taxonomy of MAS (homogene- ity, heterogeneity, communication), reasons for MAS (parallelism, scalability, intelligence), introduction to common domains (pursuit - competitive, soccer - cooperative), reactive/proactive, static/learning.

[10] - Multi-agent cooperation from control perspective. Consensus algorithms and cooperative control of multivehicle formations. Convergence of distributed consensus algorithm.

[13] - Communication in multiagent systems. Comparison of independent, fully-connected, and discrete communication. Demonstrated ability to learn an abstract communication between the agents.

[3] - Experience replay for multiagent RL. Multi-agent variant of importance sampling with natural decay for training of Q-network with results in Starcraft decoupled tasks.

[7] - Analysis of social dilemma in MARL. Discusses conflict of interests and cooperativeness in an MDP example for Prisoner’s Dilemma. Deep Q-network on a fruit Gathering game and a Wolfpack hunting game. [9] - Deep reinforcement learning with asynchronous gradient descent. An asynchronous variant of actor- critic training for half the time succeeds on a wide variety of continuous control problems including a
navigating through random 3D mazes using a visual input.

[6] - Discusses about general intelligence. Independent reinforcement learning and joint-policy correlation.
Mixtures of policies and empirical game-theoretic analysis to compute meta-strategies for policy selection in gridworld coordination games and poker.

[14] - Evaluation of cooperation and competition behavior using Deep QLearning by manipulating the rewarding function. Investigates the interaction between two learning agents in Pong and shows progression from competitive to collaborative behavior.

[11] - Multi-agent reinforcement learning under partial observability. Decentralized single-task learning that is robust to concurrent interactions of teammates. Policy transfer of distilled single-task policies into a unified policy.

[8] - Proposes an adaptation of actor-critic methods that considers action policies of other agents. Multi- agent deep RL method in simplified environment under different scenarios: Cooperative Communication, Predator-Prey, Cooperative Navigation, Physical Deception.

[4] - Cooperative Reinforcement Learning using concurrent policy learning. Policy gradient using Trust Region Policy Optimization. Comparison of dec-DQN, decDDPG, Actor-Critic and decTRPO in Pursuit, Waterworld, Multi-Walker, and Multi-Ant domains.

[1] - Complexity of environment and policy. Competitive multi-agent environment trained with self-play can produce behaviors that are far more complex than the environment itself. Discusses findings in policies in competitive environments: Run to Goal, You Shall Not Pass, Sumo, and Kick and Defend.

[5] - Deep neural networks in structured (parameterized) continuous action spaces. Extends the Deep Deterministic Policy Gradients (DDPG) into a parameterized action space within the domain of simulated RoboCup soccer.

### References
[1] T. Bansal, J. Pachocki, S. Sidor, I. Sutskever, and I. Mordatch. Emergent complexity via multi-agent competition. arXiv preprint arXiv:1710.03748, 2017.

[2] L. Busoniu, R. Babuska, and B. De Schutter. A comprehensive survey of multiagent reinforcement learning. IEEE Transactions on Systems, Man, And Cybernetics-Part C: Applications and Reviews, 38 (2), 2008, 2008.

[3] J. Foerster, N. Nardelli, G. Farquhar, T. Afouras, P. H. Torr, P. Kohli, and S. Whiteson. Stabilising experience replay for deep multi-agent reinforcement learning. arXiv preprint arXiv:1702.08887, 2017.

[4] J. K. Gupta, M. Egorov, and M. Kochenderfer. Cooperative multi-agent control using deep reinforce- ment learning. In International Conference on Autonomous Agents and Multiagent Systems, pages 66–83. Springer, 2017.

[5] M. Hausknecht and P. Stone. Deep reinforcement learning in parameterized action space. arXiv preprint arXiv:1511.04143, 2015.

[6] M. Lanctot, V. Zambaldi, A. Gruslys, A. Lazaridou, J. Perolat, D. Silver, T. Graepel, et al. A unified game-theoretic approach to multiagent reinforcement learning. In Advances in Neural Information Processing Systems, pages 4190–4203, 2017.

[7] J. Z. Leibo, V. Zambaldi, M. Lanctot, J. Marecki, and T. Graepel. Multi-agent reinforcement learn- ing in sequential social dilemmas. In Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems, pages 464–473. International Foundation for Autonomous Agents and Multiagent Systems, 2017.

[8] R. Lowe, Y. Wu, A. Tamar, J. Harb, O. P. Abbeel, and I. Mordatch. Multi-agent actor-critic for mixed cooperative-competitive environments. In Advances in Neural Information Processing Systems, pages 6379–6390, 2017.

[9] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International conference on machine learn- ing, pages 1928–1937, 2016.

[10] R. Olfati-Saber, J. A. Fax, and R. M. Murray. Consensus and cooperation in networked multi-agent systems. Proceedings of the IEEE, 95(1):215–233, 2007.

[11] S. Omidshafiei, J. Pazis, C. Amato, J. P. How, and J. Vian. Deep decentralized multi-task multi-agent reinforcement learning under partial observability. arXiv preprint arXiv:1703.06182, 2017.

[12] P. Stone and M. Veloso. Multiagent systems: A survey from a machine learning perspective. Autonomous Robots, 8(3):345–383, 2000.

[13] S. Sukhbaatar, R. Fergus, et al. Learning multiagent communication with backpropagation. In Advances in Neural Information Processing Systems, pages 2244–2252, 2016.

[14] A. Tampuu, T. Matiisen, D. Kodelja, I. Kuzovkin, K. Korjus, J. Aru, J. Aru, and R. Vicente. Multiagent cooperation and competition with deep reinforcement learning. PloS one, 12(4):e0172395, 2017.
