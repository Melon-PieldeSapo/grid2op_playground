from grid2op.Agent import MLAgent
from grid2op.Converter import ToVect

import numpy as np
import random
from collections import namedtuple, deque

from Grid2Op.grid2op.Converter.IdToAct import IdToAct
from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size (size D)
BATCH_SIZE = 64  # minibatch size (n_batch)
GAMMA = 0.99  # discount factor (gamma)
TAU = 1e-3  # for soft update of target parameters (tau)
LR = 5e-4  # learning rate (eta)
UPDATE_EVERY = 4  # how often to update the target network (C)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DqnGrid2op(MLAgent):

    def __init__(self, ENV, seed, observation_space_converter=ToVect, action_space_converter=IdToAct,
                 **kwargs_converter):
        MLAgent.__init__(self, ENV.action_space, action_space_converter, **kwargs_converter)
        self.max_reward = 0
        self.reward_range = ENV.reward_range
        self.action_space = ENV.action_space
        self.do_nothing_act = self.action_space()
        self.action_converter = action_space_converter(self.action_space)  # (64, 200, 200, 3)
        self.action_converter.seed(0)
        self.action_converter.init_converter(change_bus_vect=True, redispatch=True, curtail=False, storage=True,
                                             change_line_status=True, set_line_status=True)

        self.observation_converter = observation_space_converter(self.action_space)
        self.action_converter.seed(0)
        self.action_converter.init_converter()

        sample_obs_vect = self.observation_converter.convert_obs(ENV.reset())
        # print(sample_obs_vect)
        # sample_obs_vect.reshape(-1, len(sample_obs_vect), 1)
        # print(sample_obs_vect)

        self.state_size = len(sample_obs_vect)
        self.action_size = self.action_converter.n
        random.seed(seed)

        ##BUilding the modeL:
        # Q-Network
        units = self.state_size+self.action_size

        self.qnetwork_local = QNetwork(self.state_size, self.action_size, seed,fc1_units=units, fc2_units=units).to(device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, seed,fc1_units=units, fc2_units=units).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def convert_obs(self, observation):
        sample_obs_vect = self.observation_converter.convert_obs(observation)
        return sample_obs_vect
        # convert the observation
        # return np.concatenate((observation.load_p, observation.rho + observation.p_or))

    def convert_act(self, encoded_act):
        return self.action_converter.convert_act(int(encoded_act))

    def act(self, observation, reward, done=False):
        return self.convert_act(self._act(obs=observation, eps=100))

    def norm_reward(self,reward):
        return (reward/self.reward_range[1])*10 if reward > 0 else reward

    def _act(self, obs, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = self.convert_obs(obs)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        # ------------------- train with mini-batch sample of experiences ------------------- #
        if len(self.memory) > BATCH_SIZE:
            # If enough samples are available in memory, get random subset and learn
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        # ------------------- update target network ----------------------------------------- #
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If C (UPDATE_EVERY) steps have been reached, blend weights to the target network
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        # - qnetwork_target : apply forward pass for the whole mini-batch
        # - detach : do not backpropagate
        # - max : get maximizing action for each sample of the mini-batch (dim=1)
        # - [0].unsqueeze(1) : transform output into a flat array
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states (y)
        # - dones : detect if the episode has finished
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model (Q(Sj, Aj, w))
        # - gather : for each sample select only the output value for action Aj
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Optimize over (yj-Q(Sj, Aj, w))^2
        # * compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # * minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


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
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
