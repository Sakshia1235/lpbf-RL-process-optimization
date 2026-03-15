# Optimizing Laser Powder Bed Fusion (L-PBF) for Improved Performance: A Reinforcement Learning Approach for Choosing Process Parameters.
**Overview**

Poor choice of process parameters induces the formation of defects, which suggests the need for an improved control system to optimize the result. Hence this study focuses on the Specific goal focuses on finding optimal laser power (P) and scanning speed (V) parameters for nine different combinations, to maintain melt pool depth eventually aiming to minimize defects, enhance productivity and improve part quality. 
The core of the work involved implementing a Reinforcement Learning (RL) control policy that iteratively identified optimal laser power and scanning speed to minimize part displacement. This was supported high-fidelity melt pool simulations in Flow-3D AM and macro-scale thermal and stress analyses in Abaqus. Furthermore, I utilized Response Surface Methodology (RSM) to develop a validated surrogate model for statistical analysis and prediction of displacement values.

For our algorithm, the Q-table was initialized with all values set to 10.0 to encourage early exploration. At each step, the agent selected an action using an epsilon-greedy policy
to evaluate the learning behaviour of the Q learning agent. A plot of rewards obtained per episode was generated for hyperparameters. Learning rate (α), Discount factor (γ), Exploration rate (ε), Number of Episodes, Max steps per episode.
The hyperparameters were tweaked for 3 different configurations, the code here contains the hyperparameters for one configuration which is: -
Learning rate (α) :- 0.01
Discount factor (γ):- 0.5
Exploration rate (ε) :- 0.2 (decayed over time)
Episodes:- 50000
Max steps/episode:- 8
Epsilon decay:- 0.9995

The Q-learning method successfully converged to the optimal state-action parameter set with minimal deformations (v = 100 mm/s, P = 75 W, showing increasing V decreases P parameter trend).which aligned closely with predictions from Response Surface Methodology (RSM), demonstrating the effectiveness of the learning-based approach.
Increase in the number of episodes showed increase in convergence however it also became computationally intensive, particularly when the number of episodes was increased to 1,000,000, indicating that future implementations could benefit from more advanced and scalable methods, such as Deep Q-Networks (DQNs) integrated with neural networks or other structurally guided reinforcement learning techniques.

**Link to my thesis https://urn.fi/URN:NBN:fi-fe2025080881756**
