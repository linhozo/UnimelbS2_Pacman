# COMP90054 Project (Semester 2, 2023): PACMAN Capture the Flag

This is a group-of-three project of the Course [COMP90054 AI Planning for Autonomy](https://handbook.unimelb.edu.au/subjects/comp90054) at the University of Melbourne @ Semester 2, 2023. The purpose of this project is to implement an autonomous agent that can play the game PACMAN Capture the Flag and compete in the UoM COMP90054-2023 Pacman competition. The rules of the game can be found at [Link](http://ai.berkeley.edu/contest.html).
The main task is to develop an autonomous agent team to play PACMAN Capture the Flag by suitably modifying file myTeam.py (and maybe some other auxiliarly files students may implement).
Over the course of the project, each group must try at least least 3 AI-related techniques that have been discussed in the subject or explored by team members independently. Students can try three separate agents with different techniques, combine multiple techniques into a single agent. Some candidate techniques that students may consider are:

   * Search Algorithms (using general or domain-specific heuristic functions).
   * Classical Planning (PDDL and calling a classical planner).
   * Policy iteration or Value Iteration (Model-Based MDP).
   * Monte Carlo Tree Search or UCT (Model-Free MDP).
   * Reinforcement Learning – classical, approximate or deep Q-learning (Model-Free MDP).
   * Goal Recognition techniques (to infer intentions of opponents).
   * Game Theoretic Methods such as multi-player MCTS/reinforcement learning and backward induction.

I have chosen **Reinforcemnet Learning techniques with Q-learning and linear approximation** to develop a team of two agents, which has made to our group's final submission thanks to its best-out-of-three performance.
  * [Motivation](#motivation)
  * [Application](#application)
  * [Trade-offs](#trade-offs)     
     - [Advantages](#advantages)
     - [Disadvantages](#disadvantages)
  * [Future improvements](#future-improvements)

### Motivation  

#### Q-Learning:

For a complex and dynamic environments like Pacman Capture the Flag problem, a model-free learning technique such as Q-learning that does not require prior knowledge of the game dynamics can be considered as a highly potential solution. It encourages agents to explore different actions to discover optimal strategies and continuously update their Q-values (Q(s,a)) that represent the quality of taking a specific action in a particular state, based on new observations of the rewards that they get. This further helps agents to adapt to changing game states and opponent behaviors, which is vital in this challenge, where agents need to explore the game space to locate the foods, avoid opponents, and find safe paths.

#### Q-Learning with linear approximation:

Although Q-Learning is a promising approach, using a table to store the Q-value of each state-action pair is unfeasible due to the curse of dimensionality. It requires us to maintain a table of size |A|x|S|, which is impractically large for any non-trivial problem. In this Pacman AI problem, the states s will be the game board, which is far too high-dimensional for tabular representations. Additionally, frequent visits to every reachable state and repeated actions to accurately estimate Q(s, a) are also required. As a result, if we never visit a state s, we have no estimate of Q(s, a), even if we have visited states that are very similar to s.

To overcome these challenges, we can utilize machine learning to estimate Q-functions with linear approximation. Instead of precisely calculating the Q-function, we can generalize the quality of each state-action pair with a linear function of a set of features, each of which is assigned a suitable weight. Not only does this eliminate the need for an extensive Q-table, it also provides reasonable estimates of Q(s, a) even if we have not applied action a in state s previously.

Moreover, by implementing Q-Learning with linear approximation, we can conveniently define our attacking/defending strategies and manipulating behaviors of each of our agents in the game based on feature engineering.

Finally, it is an off-line approach, in which the agents are trained before participating in the competition. Therefore, it does not require extensive computational resource during the games.



[Back to top](#table-of-contents)

### Application

We have created two agents, each assigned the primary role of either attacker or defender. Nevertheless, they can adapt and exhibit characteristics of their teammate's main role under specific circumstances.
![image](https://github.com/linhozo/UnimelbS2_Pacman/assets/93761488/cfe86db3-68a0-42fd-ab3c-4638ddcba679)


These characteristics, pertaining to attackers and defenders, are outlined through separate sets of features and their respective weightings. The attacker's weights are determined through a combination of training and manual adjustments, while the defender's weights are manually configured. This differentiation arises due to training time constraints, and the simpler, smaller feature set for defenders, which allows distinct weight assignments based on experimentation and observation. On the other hand, attackers have a more complex feature set, making it challenging to define their weights due to potential correlations among these features.

#### Attacking Agents
##### Feature design
![image](https://github.com/linhozo/UnimelbS2_Pacman/assets/93761488/ecfaf05e-8727-406c-8815-8f439c2bab74)


##### Reward function
<body lang=EN-GB style='tab-interval:36.0pt;word-wrap:break-word'>
<!--StartFragment-->

Action | Reward
-- | --
My Agent is eaten by a ghost | -100
My Agent returns pallets and got the   score | +50*(number of pallets carried)
My Agent eats a pallet | +5
My Agent eats a capsule | +30
My Agent chooses action ‘Stop’ | -5

<!--EndFragment-->
</body>

##### Weights:

<body lang=EN-GB style='tab-interval:36.0pt;word-wrap:break-word'>
<!--StartFragment-->

Features | Initial weights | Final weights
-- | -- | --
'eaten-by-ghost' | -10000.0 | -9982.008997143268 
'dead-end-ahead' | -1000.0 | -970.2881242681565 
'stops-moving' | -100.0 | -50.02661041131593
'dist-to-nearest-ghost' | 4.0 |  8.049509252106429 
'dist-to-nearest-pallet’ | -2.0 |  2.5753564250044234
'dist-to-nearest-capsule' | -5.0 | -5.01
'dist-to-best-exit-home' | -2.0 | -3.01
'dist-to-best-entry-home' | -2.0 | -1.7071951995339094
'eats-pallet' | 200.0 | 176.90001376709728
'eats-capsule' | 300.0 | 299.99998535233027
'returns-pallet' | 400.0 | 469.7209898878529 
'attack-mode' | 1.0 | 0.9733899880879124   
'defense-mode' | -2 | -1.9919053941378122 

<!--EndFragment-->
</body>

The weights are achieved through thousands of training episodes with different approaches, including: (1) training against baseline team with a fixed layout for 2000 episodes; (2) training against baseline team with a fixed layouts in the first 1500 episodes and a random layout every 50 episodes in the next 1000 episodes; (3) training against A-star search agents for 2000 episodes; (4) training against my current Q-Learning agents with the best weights so far (self-play); (5) training against mixed teams (baseline team in the first 2000 episodes, A-star search team for the next 500 episodes, and  my current Q-Learning agents with the best weights so far in the next 500 episodes on a fixed layout.

After thousands of training episodes, it was observed that most feature weights became stable after around 800-1000 episodes. The weight for each feature was chosen from the most common ranges achieved in the abovementioned scenarios. It is noteworthy that all five scenarios were executed with the same initial weights, learning rate (0.1), discount rate (0.9) and exploration rate (0.1), from which they produced quite similar weights. The initial weights were created based on multiple attempts of experiments and observations. A bad set of initial weights resulted in inability to reach convergence. Additionally, there was only feature ‘dist-to-best-exit-home’ that had not reached convergence or the “stable point”. Therefore, the weight for this feature had been configured manually.

#### Defending Agents
##### Feature design
High level feature description:
* 'defense-mode': My Agent is at home to act as a defender after taking action
* 'stops-moving': My Agent chooses the action ‘Stop’
* 'number-of-invaders': Counting the number of observable invaders
* 'dist-to-nearest-invader': If my Agent is less than 4 distance away from the nearest invader and he has a scaredTimer of more than 0, discourage him from moving towards the invader. Otherwise, encourage him to move towards the invader.
* 'dist-to-next-pallet': If there is no invader nearby, identify the pallet that has been stolen from the previous state then find out the currently available pallet at home whose smallest distance to the stolen one. Encourage my Agent to move toward that next pallet to be guarded.
* 'dist-to-guard-position': If there is no invader nearby, and no next pallet to be guarded,  encourage my agent to move towards guard position, whose shortest average distance to door positions.
* 'reverse': My Agent is taking reversed actions.

##### Weights
<body lang=EN-GB style='tab-interval:36.0pt;word-wrap:break-word'>
<!--StartFragment-->

Features | Final weights
-- | -- 
'defense-mode' | 100 
'stops-moving' | -100 
'dist-to-guard-position' | -20 
'number-of-invaders' | -1000 
'dist-to-nearest-invader' | -10 
'dist-to-next-pallet' | -10 

<!--EndFragment-->
</body>

This set of weights were achieved through multiple experiments and observations of the agents' behaviors and reactions

### Trade-offs  
#### *Advantages*  
* **Significant reduction of memory requirements** compared to tabular Q-learning, making it feasible for large state spaces.
* **Computationally more efficient** than tabular Q-learning since the agents do not need to maintain a Q-table for each state-action pair, which can be impractical in large state spaces. No need to apply action a in state s to get a value for Q(a, s) thanks to the Q-function's ability to generalize..
* High level of freedom and creativity in terms of **feature engineering**, which can significantly impact learning efficiency and performance. Agents with certain traits and characteristics can be designed with ease.
* Linear approximators provide more **interpretable Q-value estimates**. The importance of different features can be analyzed and the agent's decision-making process can be designed with ease. Linear combinations of features can provide a reasonably accurate approximation in this AI Pacman challenge.
* No probability model is required. States can be **generalized** and defined by features and their respective weights, therefore dependencies on layout specification of the game can be reduced greatly.
* **Straightforward and simple implementation** in comparison with neural network-based approaches like deep Q-networks or simulation-based approaches like Monte-Carlo Tree Search

#### *Disadvantages*
* Linear function approximation may introduces **approximation errors** in capturing the true Q-values, especially in environments with advanced, complex opponents or layout anf nonlinear relationships in the data.
* **Designing and selecting the right features and rewards** to create an appropriate linear approximator can be challenging. It is crucial to design features and rewards that are generalized enough to well-represent most of the states but specific and expressive enough to capture different patterns and dependencies in the environment. Poor feature engineering adn reward designing can lead to suboptimal results. 
* If the feature set is too large or its logic is too complex, there's a chance that the Q-values may become **overfitted** to the training data.
* It is **greatly challenging to identify convergence** and determine how many training episodes are enough. **Extensive experiments, trainings and observations** are required to get an efficient set of initial weights that can lead to convergence or "good enough" condition.
* Given training time contraints, **manual adjustments** are required if convergence cannot be achieved, hence bias can be introduced.
* Redesigning features requires **retrainings**, which takes a lot of time and efforts.
* The selection of **hyperparameters** (e.g., learning rate, exploration rate, discount rate) is important, and finding the right values can be a non-trivial task.
* The **performance of estimators are not stable** when they play against various agent teams with random layouts, which indicates that the weights assigned for features are not accurate enough or there exists problems that are not captured by the logic in feature engineering.

[Back to top](#table-of-contents)

### Future improvements  
* **Feature engineering**: New features can be added to reflect the cooperation among our agents in the games. The current version of our team has limited cooperation between the agents. More experiments with different set of features to identify which ones contribute the most to learning can be executed as well.
* **Trainings**: More trainings can be experimented with different scenarios and hyperparamters to improve the knowledge of our agents of the environment and their behaviors, such as: training against team of attackers only; training against team of defenders only; self-play training with the same set of intital weights; training with gradually decreasing learning rate; training with slowly radomized layouts. More data such as various layout and opponents can be added in the trainings to diversify and expand the agent's knowledge.
* **Evaluation**: Different metrics such as accumualted rewards can be utilized to measure the performance and identify convergence or "good enough" state. Conitnuously benchmarking our agents' perfomrance with other agents using different techniques such as A-star heuristic search, MCTS or deep reinforcement learning models. 
* **Reward Shaping**: Reward shaping can be used to guide the learning process.
* **Combining multiple techniques**: more experiments with combination of MCTS and Q-learning with linear approximation can be conducted to leverage each techniques' advantages where Q-learning can provide a promising guide for MCTS for its simulation and exapnasion process.

[Back to top](#table-of-contents)
***

