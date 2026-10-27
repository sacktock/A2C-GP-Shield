import jax.random as jr
import jax.numpy as jnp
from jax import jit
import jax
import gym
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.linalg import inv

@jit
def matrix_transpose(batched_matrix):
    assert len(batched_matrix.shape) == 3
    return jnp.transpose(batched_matrix, axes=(0, 2, 1))

@jit
def squash_sin(mu, cov, action_limit):
    k = mu.shape[1]
    action_limits = jnp.ones((1, k)) * action_limit

    squashed_mu  = action_limits * jnp.exp(- jnp.diag(cov) / 2) * jnp.sin(mu)

    lq = -(jnp.diag(cov)[:, None] + jnp.diag(cov)[None, :]) / 2
    q = jnp.exp(lq)

    squashed_cov = (jnp.exp(lq + cov) - q) * jnp.cos(jnp.transpose(mu) - mu) \
        - (jnp.exp(lq - cov) - q) * jnp.cos(jnp.transpose(mu) + mu)
    squashed_cov = action_limits * jnp.transpose(action_limits) * squashed_cov / 2

    cross_cov = action_limits * jnp.diag((jnp.exp(-jnp.diag(cov) / 2) * jnp.cos(mu))[0, :])
    return squashed_mu, squashed_cov, cross_cov

class Agent(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def act(self, key, obs):
        return NotImplementedError

    @property
    def policy(self):
        return lambda k, x: self.act(k, x)

class JAXAgent(Agent):

    @property
    def policy(self):
        return lambda k, x: jax.device_get(self.act(k, x).block_until_ready())


class LQR_Agent(JAXAgent):

    def __init__(self, A, B, Q, R, action_limit=1.0):
        super().__init__()

        if Q is None:
            self.Q = np.eye(A.shape[0])
        else:
            self.Q = Q

        if R is None:
            self.R = np.ones(B.shape[1]).reshape(-1, 1)
        else:
            self.R = R

        assert A.shape == Q.shape
        assert R.shape[0] == B.shape[1]

        P = solve_continuous_are(A, B, Q, R)

        # Compute the LQR gain
        K = inv(R) @ B.T @ P
        self.K = jnp.array(K)

        self.action_limit = action_limit
        print(f"Constructed linear controller K with shape {self.K.shape}")
        print(self.K)

    def act(self, key, obs):
        return self._act(self.K, obs, self.action_limit)

    def propagate(self, obs, cov):
        return self._propagate(self.K, obs, cov, self.action_limit)

    @staticmethod
    @jit
    def _act(K, obs, action_limit):
        cov = jnp.eye(K.shape[1]) * 1e-6
        next_mu = - K @ obs
        next_mu = next_mu.reshape(1, -1)
        next_cov = (-K @ cov) @ -K.T
        next_mu, _, _ = squash_sin(next_mu, next_cov, action_limit)
        return jnp.squeeze(next_mu, axis=0)

    @staticmethod
    @jit
    def _propagate(K, obs, cov, action_limit):
        next_mu = -K @ obs
        next_mu = next_mu.reshape(1, -1)
        next_cov = (-K @ cov) @ -K.T
        cross_cov_1 = -K.T
        next_mu, next_cov, cross_cov_2 = squash_sin(next_mu, next_cov, action_limit)
        cross_cov = cross_cov_1 @ cross_cov_2
        return jnp.squeeze(next_mu, axis=0), next_cov, cross_cov

class Linear_Agent(JAXAgent):

    def __init__(self, key, input_dim, output_dim, action_limit=1.0, **kwargs):
        super().__init__()
        if "weights" in kwargs:
            self.weights = jnp.array(kwargs["weights"])
            assert self.weights.shape == (output_dim, input_dim)
            if "input_bias" in kwargs:
                self.input_bias = jnp.array(kwargs["input_bias"])
                assert self.input_bias.shape == (input_dim,)
            else:
                self.input_bias = jnp.zeros((input_dim,))
            if "output_bias" in kwargs:
                self.output_bias = jnp.array(kwargs["output_bias"])
                assert self.output_bias.shape == (output_dim,)
            else:
                self.output_bias = jnp.zeros((output_dim,))
        else:
            self.weights = jr.normal(key, shape=(output_dim, input_dim))
            self.input_bias = jnp.zeros((input_dim,))
            self.output_bias = jr.normal(key, shape=(output_dim,))

        self.action_limit = action_limit

    def act(self, key, obs):
        return self._act(self.weights, self.input_bias, self.output_bias, obs, self.action_limit)

    def propagate(self, obs, cov):
        return self._propagate(self.weights, self.input_bias, self.output_bias, obs, cov, self.action_limit)

    def act_dim(self, obs):
        return self._act_dim(self.weights, self.input_bias, self.output_bias, obs, self.action_limit)

    @staticmethod
    @jit
    def _act_dim(W, b_in, b_out, obs, action_limit):
        dim = obs.shape[0]
        cov = jnp.tile(jnp.eye(W.shape[1]) * 1e-6, (dim, 1, 1))
        next_mu = (W @ (obs - b_in.reshape(1, -1)).T + b_out.reshape(1, -1).T).T
        next_cov = matrix_transpose(cov @ W.T) @ W.T
        k = next_mu.shape[1]
        action_limits = jnp.ones((1, k)) * action_limit
        squashed_mu  = action_limits * jnp.exp(- jnp.diagonal(next_cov, axis1=1, axis2=2) / 2) * jnp.sin(next_mu)
        return squashed_mu

    @staticmethod
    @jit
    def _act(W, b_in, b_out, obs, action_limit):
        cov = jnp.eye(W.shape[1]) * 1e-6
        next_mu = (jnp.dot(W, (obs-b_in)) + b_out).reshape(1, -1)
        next_cov = jnp.dot(jnp.dot(W, cov), jnp.transpose(W))
        next_mu, _, _ = squash_sin(next_mu, next_cov, action_limit)
        return jnp.squeeze(next_mu, axis=0)

    @staticmethod
    @jit
    def _propagate(W, b_in, b_out, obs, cov, action_limit):
        next_mu = (jnp.dot(W, (obs-b_in)) + b_out).reshape(1, -1)
        next_cov = jnp.dot(jnp.dot(W, cov), jnp.transpose(W))
        cross_cov_1 = jnp.transpose(W)
        next_mu, next_cov, cross_cov_2 = squash_sin(next_mu, next_cov, action_limit)
        cross_cov = jnp.matmul(cross_cov_1, cross_cov_2)
        return jnp.squeeze(next_mu, axis=0), next_cov, cross_cov

class Random_Agent(JAXAgent):

    def __init__(self, action_space, continuous=True):
        super().__init__()
        self.continuous = continuous
        if self.continuous:
            self.action_space = action_space
            assert isinstance(self.action_space, gym.spaces.Box)
        else:
            self.n_actions = action_space.n

    def act(self, key, obs):
        if self.continuous:
            return jr.uniform(key, minval=self.action_space.low, maxval=self.action_space.high, shape=self.action_space.shape)
        else:
            return jr.choice(key, self.n_actions).item()



