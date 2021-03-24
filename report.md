## Multi-Agent Deep Deterministic Policy Gradients (MADDPG)

This environment requires multiple agents to play against each other in continuous space. In the previous project, ddpg's were an apt choice for solving the
environment which was also a continuous space environment.
The MADDPG is a multi agent ddpg where every agent has an observation space and continuous action space. <br>
Also, each agent has three components:
- An actor-network that uses local observations for deterministic actions
- A target actor-network with identical functionality for training stability
- A critic-network that uses joint states action pairs to estimate Q-values

The agents in the MADDPG share actor and critic models. The actor model learns to predict an action vector given the state of the environment 
as observed by the agent. The critic model learns Q-values for combined states and actions from all the agents in the environment. 
In this way, the actor only relies on local information while the critic uses global information.
MADDPG uses an experience replay for efficient off-policy training. At each timestep, the agent stores the transition,
where we store the joint state, next joint state, joint action, and each of the agents’ received rewards. 
Then, we sample a batch of these transitions from the experience replay to train our agent.

## Hyperparameters
The hyperparameters used for the algorithm are:
- seed = 42
- actor_hidden_units = (256,128)
- actor_learning_rate = 3e-4
- critic_hidden_units = (1024,512)
- critic_learning_rate = 3e-4
- weight_decay = 0
- shared_replay_buffer = True
- batch_size = 128
- buffer_size = int(1e6)
- discount = 0.99
- update_every = 4
- tau = 6e-2

## Components
Moreover a Ornstein-Uhlenbeck was used for adding noise and Replay Buffer was used to store shared experiences for both actors. 

The architecture of the target and local networks for the Actor and Critic models are shown below:
![image](https://user-images.githubusercontent.com/46076665/112245985-59511a80-8c28-11eb-8f76-a354d31d3e07.png)

In these models, a batch normalizer with momentum of 0.1 and epsilon 1e-5 was used and dropout of 0.3. 
The input layer is the state size and the output layer of the actor models are the action size. The critic models have an output layer of 1.


## Results
![image](https://user-images.githubusercontent.com/46076665/112246373-ec8a5000-8c28-11eb-92b7-5bd885f5f7be.png)

The maddpg was able to solve the environment within 800 episodes

## Improvements

This can be improved by:
- using better actor and critic models
- using a prioritized experience replay buffer so as to replay important transitions more frequently, and therefore learn more efficiently
- hyperparameter tuning
