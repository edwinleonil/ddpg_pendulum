import numpy as np
import random
import copy
from collections import namedtuple, deque

from Generic_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

# TUNING THESE PARAMETERS ARE VERY IMPORTANT

BUFFER_SIZE = int(1e5)  # replay buffer size - 1e5
BATCH_SIZE = 64        # minibatch size - small size overfit and too big that overgeneralise and dont fit new data 128
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    # Interacts with and learns from the environment
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        """ Initialize local networks """
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)  
        
        """ Initialize target networks """
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        """ Initialize replay buffer R"""
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed) # Replay memory
    
    def step(self, state, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE: 
            experiences = self.memory.sample() # get a random sample of (s_t,a_t,r_t,s_t+1) from R
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -2, 2)

    def reset(self):
        self.noise.reset()

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

        """ ---------------------------- Update critic ---------------------------- """
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        """"Compute Q_targets (y_i = r_i + gama*Q_target_next)"""
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute Q_expected
        Q_expected = self.critic_local(states, actions)
        """Compute critic loss (critic_loss = Q_expected - Q_target)"""
        self.critic_optimizer.zero_grad()
        critic_loss = F.mse_loss(Q_targets, Q_expected)        
        critic_loss.backward()  # backpropagate to optimise parameters
        self.critic_optimizer.step()

        """ ---------------------------- Update actor policy---------------------------- """
        
        """Compute actor loss (actor_loss = - critic_local(states, actions)"""
        self.actor_optimizer.zero_grad()
        actions_pred = self.actor_local(states)
        # Compute actor loss - (make critic_local() negative so is just the oposity to maximazing the Q values (Q_expected))
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss       
        # backpropagate to optimise parameters
        actor_loss.backward()
        self.actor_optimizer.step()

        """ ----------------------- Update target networks ----------------------- """
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        Target_model = τ*Local_model + (1 - τ)*Target_model
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

class OUNoise:
    # Ornstein-Uhlenbeck process
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2): # tetha was 0.15
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        # Reset the internal state (= noise) to mean (mu)
        self.state = copy.copy(self.mu)

    def sample(self):
        # Update internal state and return it as a noise sample
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    # Fixed-size buffer to store experience tuples
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """ Store transition (s_t,a_t,r_t,s_t+1) in R(memory) """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Sample a randomly minibatch of N transition (s_i,a_i,r_i,s_i+1) from R(memory) """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        # Return the current size of internal memory
        return len(self.memory)