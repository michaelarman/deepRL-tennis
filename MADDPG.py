import torch
import os
import copy
from collections import namedtuple, deque
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import random

# BATCH_SIZE = 128        # minibatch size
# BUFFER_SIZE = int(1e6)  # replay buffer size
# GAMMA = 0.99            # discount factor
# LR_ACTOR = 3e-3         # learning rate of the actor 
# LR_CRITIC = 3e-3        # learning rate of the critic
# TAU = 6e-2              # for soft update of target parameters
# WEIGHT_DECAY = 0        # L2 weight decay
# UPDATE_EVERY = 4        # time steps between network updates
# N_UPDATES = 1           # number of times training
# ALPHA = 0               # PER 1 = full prioritization, 0 = no prioritization
# BETA_START = 0          # PER - What degree to use importance weights (0 - no corrections, 1 - full correction)

# ADD_NOISE = True
# NOISE_START = 6
# NOISE_END = 0
# NOISE_DECAY = 250

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG:
    def __init__(self, config):
        self.config = config
        
        if config.shared_replay_buffer:
            self.memory = config.memory_fn()
            self.config.memory = self.memory
        
        self.ddpg_agents = [DDPGAgent(self.config) for _ in range(config.num_agents)]
        
        self.t_step = 0
     
    def reset(self):
        for agent in self.ddpg_agents:
            agent.reset()
    
    def act(self, all_states):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(np.expand_dims(states, axis=0)) for agent, states in zip(self.ddpg_agents, all_states)]
        return actions
    
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.config.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.config.batch_size:
                for agent in self.ddpg_agents:
                    if self.config.shared_replay_buffer:
                        experiences = self.memory.sample()
                    else:
                        experiences = agent.memory.sample()
                    
                    agent.learn(experiences, self.config.discount)

class DDPGAgent:
    def __init__(self, config):
        self.config = config
        self.seed = config.seed
        
        # Actor Network (w/ Target Network)
        self.actor_local = config.actor_network_fn()
        self.actor_target = config.actor_network_fn()
        self.actor_optimizer = config.actor_optimizer_fn(self.actor_local.parameters())
        
        # Critic Network (w/ Target Network)
        self.critic_local = config.critic_network_fn()
        self.critic_target = config.critic_network_fn()
        self.critic_optimizer = config.critic_optimizer_fn(self.critic_local.parameters())
        
        # ----------------------- initialize target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, 1)
        self.soft_update(self.actor_local, self.actor_target, 1)

        self.noise = config.noise_fn()
        
        if config.shared_replay_buffer:
            self.memory = config.memory
        else:
            self.memory = config.memory_fn()
        
    def reset(self):
        self.noise.reset()
        
    def act(self, states):            
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(self.config.device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        actions += self.noise.sample()
        return np.clip(actions, -1, 1)
                
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.config.tau)
        self.soft_update(self.actor_local, self.actor_target, self.config.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)
        
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplay:
    """Fixed-size buffer to store prioritized experience tuples."""

    def __init__(self, batch_size, buffer_size, seed, alpha, device):
        """Initialize a ReplayBuffer object."""

        self.memory = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.device = device
        self.epsilon = 1e-5       
          
    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.priorities.append(priority)
    
    def sample(self, beta):
        """Randomly sample a batch of experiences from memory."""
        priorities = np.array(self.priorities).reshape(-1)
        priorities = np.power(priorities + self.epsilon, self.alpha)  # add a small value epsilon to ensure numeric stability
        p = priorities/np.sum(priorities)  # compute a probability density over the priorities
        sampled_indices = np.random.choice(np.arange(len(p)), size=self.batch_size, p=p)  # choose random indices given p
        experiences = [self.memory[i] for i in sampled_indices]     # subset the experiences
        p = np.array([p[i] for i in sampled_indices]).reshape(-1)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        weights = np.power(len(experiences) * p, -beta)
        weights /= weights.max()
        weights = torch.from_numpy(weights).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def update(self, indices, priorities):
        """Update the priority values after training given the samples drawn."""
        for i, priority in zip(indices, priorities):
            self.priorities[i] = priority

    def ready(self):
        """Return the current size of internal memory."""
        return len(self.memory) >= self.batch_size

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.13, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size, state_size, hidden_units, seed, gate=torch.relu, final_gate=torch.tanh):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_units (array): Number of nodes for layers
            seed (int): Random seed
            gate (function): activation function
            final_gate (function): final activation function
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.gate = gate
        self.final_gate = final_gate
        self.normalizer = nn.BatchNorm1d(state_size)
        dims = (state_size, ) + hidden_units
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.output = nn.Linear(dims[-1], action_size)
        self.reset_parameters()

    def reset_parameters(self):        
        for layer in self.layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, states):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.normalizer(states)
        for layer in self.layers:
            x = self.gate(layer(x))
        return self.final_gate(self.output(x))

    
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, action_size, state_size, hidden_units, seed, gate=torch.relu, dropout=0.3):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_units (array): Number of nodes for layers
            seed (int): Random seed
            gate (function): activation function
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.gate = gate
        self.dropout = nn.Dropout(p=dropout)
        self.normalizer = nn.BatchNorm1d(state_size)
        dims = (state_size, ) + hidden_units
        self.layers = nn.ModuleList()
        count = 0
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            if count == 1:
                self.layers.append(nn.Linear(dim_in+action_size, dim_out))
            else:
                self.layers.append(nn.Linear(dim_in, dim_out))
            count += 1
        self.output = nn.Linear(dims[-1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.normalizer(states)
        xs = self.gate(self.layers[0](xs))
        x = torch.cat((xs, actions), dim=1)
        for i in range(1, len(self.layers)):
            x = self.gate(self.layers[i](x))
        x = self.dropout(x)
        return self.output(x)