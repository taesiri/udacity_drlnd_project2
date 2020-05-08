import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy 
import math
from collections import namedtuple, deque
from torch.utils.data import TensorDataset, DataLoader

BUFFER_SIZE  = int(2e5)  # replay buffer size
BATCH_SIZE   = 128       # minibatch size
GAMMA        = 0.999     # discount factor
TAU1         = 1e-3      # for soft update of target parameters
TAU2         = 1e-3      # for soft update of target parameters
LR_ACTOR     = 2e-4      # learning rate 
LR_CRITIC    = 2e-4      # learning rate 
LR           = 2e-4      # learning rate 
UPDATE_EVERY = 4         # how often to update the network
PI = math.pi             # 3.1415...
ENTROPY_BETA = 1e-4
PPO_UPDATES  = 4         # Number of PPO updates per Step
ENTROPY_BETA = 0.001     # Entropy Multiplier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPQAgent():
  """Interacts with and learns from the environment."""

  def __init__(self, state_size, action_size, actor_network_local, actor_network_target, critic_network_local, critic_network_target, seed=0):
    """ Deep Deterministic Policy Gradient Agent - Initialize an Agent object.
    
    Params
    ======
      state_size (int): dimension of each state
      action_size (int): dimension of each action
      seed (int): random seed
    """
    self.state_size = state_size
    self.action_size = action_size
    self.seed = random.seed(seed)
    
    # Actor-Critic Networks, Local and Target
    self.actor_local   = actor_network_local
    self.actor_target  = actor_network_target
    self.critic_local  = critic_network_local
    self.critic_target = critic_network_target
    self.optimizerActor = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
    self.optimizerCritic = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)
    self.train_mode = True

    # Replay memory
    self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
    # Initialize time step (for updating every UPDATE_EVERY steps)
    self.t_step = 0
    # Testing Training Separation
    self.Testing = False
  
  def step(self, state, action, reward, next_state, done):
    # Save experience in replay memory
    self.memory.add(state, action, reward, next_state, done)
    
    # Learn every UPDATE_EVERY time steps.
    self.t_step = (self.t_step + 1) % UPDATE_EVERY
    if self.t_step == 0:
      # If enough samples are available in memory, get random subset and learn
      if len(self.memory) > BATCH_SIZE:
        experiences = self.memory.sample()
        self.learn(experiences, GAMMA)

  def act(self, state, eps=0.):
    """Returns actions for given state as per current policy.
    
    Params
    ======
      state (array_like): current state
      eps (float): epsilon, for epsilon-greedy action selection
    """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    self.actor_local.eval() # Set to Evaluation Mode
    with torch.no_grad(): # Dont Store Gradients
      action = self.actor_local(state)
    self.actor_local.train() # Set Back to Train Mode

    action = action.cpu().numpy()
    # Add noise for exploration, No need to add OU noise, as the Normal noise can do the job.
    # This should be disabled in Testing
    noise = np.random.normal(0, 1 , action.shape)
    if self.Testing:
      noise = np.zeros_like(noise);

    return action+noise

  def learn(self, experiences, gamma):
    """Update value parameters using given batch of experience tuples.

    Params
    ======
      experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
      gamma (float): discount factor
    """
    state_batch, action_batch, reward_batch, next_states_batch, doneMask = experiences
    
    # ------------------- Critic Loss ------------------- #
    state_action_values = self.critic_local(state_batch, action_batch)          # Q(S, a)
    next_actions = self.actor_target(next_states_batch)                         # Actions for next state (Mu(S'))
    next_states_values = self.critic_target(next_states_batch, next_actions)    # Value of Selected Action for the Next State Q(S', Mu(S'))
    y = reward_batch + (gamma * (next_states_values * doneMask))
    loss_critic = F.mse_loss(state_action_values, y)

    # ------------------- Actor Loss ------------------- #
    actions_from_policy = self.actor_local(state_batch)                         # Mu(S)
    loss_actor = self.critic_local(state_batch, actions_from_policy)            # Q(S, Mu(S))
    loss_actor = -loss_actor.mean()
    
    # ------------------- Optimize Both Models ------------------- #

    # Actor
    self.optimizerActor.zero_grad()
    loss_actor.backward()
    for param in self.actor_local.parameters():
      param.grad.data.clamp_(-1, 1)               # Gradient cliping
    self.optimizerActor.step()
    
    # Critic
    self.optimizerCritic.zero_grad() 
    loss_critic.backward()
    for param in self.critic_local.parameters():
      param.grad.data.clamp_(-1, 1)               # Gradient cliping
    self.optimizerCritic.step()
    
    # ------------------- update target network ------------------- #
    self.soft_update(self.critic_local, self.critic_target, TAU1)                     
    self.soft_update(self.actor_local, self.actor_target, TAU2)                     

  def soft_update(self, local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
      local_model (PyTorch model): weights will be copied from
      target_model (PyTorch model): weights will be copied to
      tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class VPGAgent():
  """Interacts with and learns from the environment."""

  def __init__(self, state_size, action_size, policy_network, value_network, seed=0):
    """ REINFORCE Agent - Initialize an Agent object.
    
    Params
    ======
      state_size (int): dimension of each state
      action_size (int): dimension of each action
      policy_network (FullyConnectedPolicy): policy network
      seed (int): random seed
    """
    self.state_size = state_size
    self.action_size = action_size
    self.seed = random.seed(seed)
    
    self.policy_network   = policy_network
    self.value_network    = value_network
    self.optimizerPolicy  = optim.Adam(self.policy_network.parameters(), lr=LR_ACTOR)
    self.optimizerCritic   = optim.Adam(self.policy_network.parameters(), lr=LR_CRITIC)
    self.t_step = 0

  def act(self, state):
    """Returns actions for given state as per current policy.
    
    Params
    ======
      state (array_like): current state
      eps (float): epsilon, for epsilon-greedy action selection
    """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    self.policy_network.eval()  # Set to Evaluation Mode
    mus, sigmas = None, None
    with torch.no_grad():       # Dont Store Gradients
      mus, sigmas = self.policy_network(state)
    self.policy_network.train() # Set Back to Train Mode

    mus    = mus.cpu().numpy()
    sigmas = sigmas.cpu().numpy()

    actions = np.random.normal(mus, sigmas, mus.shape)
    actions = np.clip(actions, -1, 1)

    return actions

  def compute_log_vars(self, x, mu , sigma):
      """ Compute Log of PDF of Normal Distribution """
      return -((x - mu)**2 / (2*(sigma**2).clamp(min=1e-3))) - torch.log(torch.sqrt(2*PI*sigma**2))

  def learn(self, states, actions, rewards, gamma):
    """Update value parameters using given batch of experience tuples.
    NO BATCHING JUST YET!
    Params
    ======
      states  (list):
      actions (list):
      rewards (list): rewards at each time step 
      gamma  (float): discount factor
    """

    # Compute Discounted Rewards and Normalized Rewards
    discount = gamma**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    mean = np.mean(rewards_future)
    std = np.std(rewards_future) + 1.0e-10
    rewards_normalized = (rewards_future - mean)/std
    drewards = torch.tensor(rewards_normalized).float().to(device)


    states = torch.tensor(states).float().to(device)
    values = self.value_network(states).detach()
    actions = torch.tensor(actions).float().to(device)
    mus, sigmas = self.policy_network(states)

    log_actions = self.compute_log_vars(actions, mus, sigmas)
    # Compute Loss

    delta = (drewards-values.squeeze())

    policy_loss = -(delta.unsqueeze(1)*log_actions).mean()
    entropy = torch.log(torch.sqrt(2*PI*sigmas**2)).mean()
    policy_loss =  policy_loss + (ENTROPY_BETA*entropy)

    values = self.value_network(states)
    loss_critic = F.mse_loss(values.squeeze(), drewards)
    # print(loss_critic)

    # ------------------- Optimize The Model ------------------- #

    self.optimizerPolicy.zero_grad()
    policy_loss.backward()
    for param in self.policy_network.parameters():
      param.grad.data.clamp_(-1, 1)               # Gradient cliping
    self.optimizerPolicy.step()

    self.optimizerCritic.zero_grad()
    loss_critic.backward()
    for param in self.value_network.parameters():
      param.grad.data.clamp_(-1, 1)               # Gradient cliping
    self.optimizerCritic.step()

class PPOAgent():
  """Interacts with and learns from the environment."""

  def __init__(self, state_size, action_size, actor_critic_network, seed=0):
    """ PPO Agent - Initialize an Agent object.
    
    Params
    ======
      state_size (int): dimension of each state
      action_size (int): dimension of each action
      actor_critic_network (FCActorCriticBrain): Actor Critic Network
      seed (int): random seed
    """
    self.state_size  = state_size
    self.action_size = action_size
    self.seed        = random.seed(seed)
    
    self.actor_critic_network   = actor_critic_network
    self.optimizer              = optim.Adam(self.actor_critic_network.parameters(), lr=LR)

  def act(self, state):
    """Returns actions for given state as per current policy.
    
    Params
    ======
      state (array_like): current state
    """
    state = torch.tensor(state).float().to(device)
    action_distribution, value = self.actor_critic_network(state)

    action = action_distribution.sample()
    log_prob = action_distribution.log_prob(action)
    entropy =  action_distribution.entropy().mean()

    return action, log_prob, entropy, value

  def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    """Computes GAE """
    gae = 0
    values = values + [next_value]
    returns = []
    for step in reversed(range(len(rewards))):
      delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
      gae = delta + gamma * tau * masks[step] * gae
      returns.insert(0, gae + values[step])
    return returns

  def ppo_optimize(self, state, action, old_log_probs, return_, advantage, eps=0.2):
    """learn - Performs One step gradient descent on a batch of data"""
    dist, value = self.actor_critic_network(state)
    entropy = dist.entropy().mean()
    new_log_probs = dist.log_prob(action)

    ratio = (new_log_probs - old_log_probs).exp()
    surr = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantage

    actor_loss  = - torch.min(ratio * advantage, surr).mean()
    critic_loss = (return_ - value).pow(2).mean()

    loss = 0.5 * critic_loss + actor_loss - ENTROPY_BETA * entropy

    # ------------------- Optimize the Models ------------------- #
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()   

  def learn(self, states, actions, log_probs, returns, advantages):
    """PPO Update - Loop through all collected trajectories and Update the Network."""

    dataset = TensorDataset(states, actions, log_probs, returns, advantages)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    for _ in range(PPO_UPDATES):
      for batch_idx, (state, action, old_log_probs, return_, advantage) in enumerate(loader):
        self.ppo_optimize(state, action, old_log_probs, return_, advantage)

class ReplayBuffer:
  """Fixed-size buffer to store experience tuples."""

  def __init__(self, action_size, buffer_size, batch_size, seed):
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
  
  def add(self, state, action, reward, next_state, done):
    """Add a new experience to memory."""
    e = self.experience(state, action, reward, next_state, done)
    self.memory.append(e)
  
  def sample(self):
    """Randomly sample a batch of experiences from memory."""
    experiences = random.sample(self.memory, k=self.batch_size)

    states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
    actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
    rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
    next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
    dones = torch.from_numpy(np.vstack([1-e.done for e in experiences if e is not None])).float().to(device)

    return (states, actions, rewards, next_states, dones)

  def __len__(self):
    """Return the current size of internal memory."""
    return len(self.memory)