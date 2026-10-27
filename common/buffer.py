
import gym
import numpy as np
from collections import defaultdict

class BaseBuffer:

    def __init__(self, buffer_size, observation_space, action_space):
        assert type(buffer_size) is int
        assert isinstance(observation_space, gym.spaces.Box)
        assert isinstance(action_space, gym.spaces.Box) or isinstance(action_space, gym.spaces.Discrete)

        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space

        self.pos = 0
        self.full = False

        if isinstance(self.action_space, gym.spaces.Box):
            self.action_dim = int(np.prod(self.action_space.shape))
        elif isinstance(self.action_space, gym.spaces.Discrete):
            self.action_dim = int(self.action_space.n)
        else:
            raise NotImplementedError

        self.input_shape = self.observation_space.shape

        self.reset()

    def reset(self):
        self.pos = 0
        self.full = False

    def add(self):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

class TransitionBuffer(BaseBuffer):

    def __init__(self, buffer_size, observation_space, action_space):
        super().__init__(buffer_size, observation_space, action_space)

    def reset(self):
        super().reset()
        self.states = np.zeros((self.buffer_size, *self.input_shape), dtype=np.float64)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float64)
        self.next_states = np.zeros((self.buffer_size, *self.input_shape), dtype=np.float64)
        
    def add(self, new_states, new_actions, new_next_states):

        new_states = np.array(new_states, dtype=np.float64)
        new_actions = np.array(new_actions, dtype=np.float64)
        new_next_states = np.array(new_next_states, dtype=np.float64)

        new_actions = new_actions.reshape(new_actions.shape[0], self.action_dim)

        assert new_states.shape[1:] == self.input_shape
        assert new_actions.shape[1] == self.action_dim
        assert new_next_states.shape[1:] == self.input_shape

        assert len(new_states) == len(new_actions) == len(new_next_states)

        indices = [pos % self.buffer_size for pos in range(self.pos, self.pos + len(new_states))]

        self.pos = self.pos + len(new_states)

        if self.pos >= self.buffer_size:
            self.pos = self.pos % self.buffer_size
            self.full = True

        self.states[indices] = new_states
        self.actions[indices] = new_actions
        self.next_states[indices] = new_next_states

    def get(self):
        if self.full:
            return self.states, self.actions, self.next_states
        else:
            return self.states[:self.pos], self.actions[:self.pos], self.next_states[:self.pos]

class StateBuffer(BaseBuffer):

    def __init__(self, buffer_size, observation_space, action_space):
        super().__init__(buffer_size, observation_space, action_space)

    def reset(self):
        self.states = np.zeros((self.buffer_size, *self.input_shape), dtype=np.float64)

    def add(self, new_states):

        new_states = np.array(new_states, dtype=np.float64)

        assert new_states.shape[1:] == self.input_shape

        batch_size = len(new_states)

        indices = [pos % self.buffer_size for pos in range(self.pos, self.pos + len(new_states))]

        self.pos = self.pos + len(new_states)

        if self.pos >= self.buffer_size:
            self.pos = self.pos % self.buffer_size
            self.full = True

        self.states[indices] = new_states

    def get(self):
        if self.full:
            return self.states
        else:
            return self.states[:self.pos]

    def sample(self, batch_size, **kwargs):
        n = self.buffer_size if self.full else self.pos
        indices = np.random.choice(n, size=batch_size)
        return self.states[indices], np.ones(batch_size)


class PrioritizedStateBuffer(StateBuffer):

    def __init__(self, buffer_size, observation_space, action_space, gamma=0.99, alpha=0.6):
        super().__init__(buffer_size, observation_space, action_space)
        self.gamma = gamma
        self.alpha = alpha

    def reset(self):
        super().reset()
        self.returns = np.zeros((self.buffer_size), dtype=np.float64)
        self.dones = np.zeros((self.buffer_size), dtype=np.float64)

    def add(self, new_states, new_rewards, new_dones):

        new_states = np.array(new_states, dtype=np.float64)
        new_rewards = np.array(new_rewards, dtype=np.float64)
        new_dones = np.array(new_dones, dtype=np.float64)

        assert new_states.shape[1:] == self.input_shape

        assert len(new_states) == len(new_rewards) == len(new_dones)

        batch_size = len(new_states)

        new_returns = np.zeros(batch_size, dtype=np.float64)

        running_return = 0.0

        for step in reversed(range(batch_size)):
            if new_dones[step]:
                running_return = 0.0
            running_return = new_rewards[step] + self.gamma * running_return
            new_returns[step] = running_return 

        indices = [pos % self.buffer_size for pos in range(self.pos, self.pos + len(new_states))]

        self.pos = self.pos + len(new_states)

        if self.pos >= self.buffer_size:
            self.pos = self.pos % self.buffer_size
            self.full = True

        self.states[indices] = new_states
        self.returns[indices] = new_returns
        self.dones[indices] = new_dones

        t = 0
        step = indices[0]
        running_return = self.returns[step]
        
        if self.full:
            while True:
                step = (step - 1) % self.buffer_size
                t = t + 1
                if self.dones[step] or step == indices[-1]:
                    break
                self.returns[step] = self.returns[step] + (self.gamma**t)*running_return
        else:
            while True:
                step = step - 1
                t = t + 1
                if step < 0 or self.dones[step]:
                    break
                self.returns[step] = self.returns[step] + (self.gamma**t)*running_return

    def get(self):
        if self.full:
            return self.states, self.returns
        else:
            return self.states[:self.pos], self.returns[:self.pos]

    def sample(self, batch_size, beta=0.4, **kwargs):

        if self.full:
            prios = self.returns - np.min(self.returns)
        else:
            prios = self.returns[:self.pos] - np.min(self.returns[:self.pos])

        probs = (prios + 1e-6) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(prios), size=batch_size, p=probs)

        if self.full:
            total = self.buffer_size
        else:
            total = self.pos

        weights = (total * probs[indices]) ** -beta
        weights /= np.max(weights)

        weights = np.array(weights, dtype=np.float64)

        return self.states[indices], weights

class NoveltyStateBuffer(StateBuffer):

    def __init__(self, buffer_size, observation_space, action_space, n_bins=32, alpha=0.6):
        super().__init__(buffer_size, observation_space, action_space)
        self.n_bins = n_bins
        self.alpha = alpha
        self.bin_edges = np.array([np.linspace(low, high, n_bins + 1) for low, high in zip(self.observation_space.low, self.observation_space.high)], dtype=np.float64)

    def reset(self):
        super().reset()
        self.bins = np.zeros((self.buffer_size, *self.input_shape), dtype=np.int64)
        self.bin_counts = defaultdict(int)

    def _discretize(self, state):
        return np.array([np.digitize(state[i], self.bin_edges[i]) -1 for i in range(len(self.bin_edges))], dtype=np.int64)

    def add(self, new_states):

        new_states = np.array(new_states, dtype=np.float64)

        new_states = np.clip(new_states, self.observation_space.low, self.observation_space.high)

        assert new_states.shape[1:] == self.input_shape

        new_state_bins = np.array([self._discretize(state) for state in new_states], dtype=np.int64)

        indices = [pos % self.buffer_size for pos in range(self.pos, self.pos + len(new_states))]

        self.pos = self.pos + len(new_states)

        if self.pos >= self.buffer_size:
            self.pos = self.pos % self.buffer_size
            self.full = True

        self.states[indices] = new_states

        self.bins[indices] = new_state_bins

        for b in new_state_bins:
            self.bin_counts[tuple(b)] += 1

    def sample(self, batch_size, beta=0.4, **kwargs):

        if self.full:
            bins = self.bins
        else:
            bins = self.bins[:self.pos]

        counts = np.array([self.bin_counts[tuple(b)] for b in bins])
        probs = 1.0 / (counts + 1e-6) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(probs), size=batch_size, p=probs)

        if self.full:
            total = self.buffer_size
        else:
            total = self.pos

        weights = (total * probs[indices]) ** -beta
        weights /= np.max(weights)

        weights = np.array(weights, dtype=np.float64)

        return self.states[indices], weights







        