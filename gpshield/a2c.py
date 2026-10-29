from gpshield.agents import JAXAgent
from dataclasses import field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import flax.linen as nn
import jax
from jax.tree_util import tree_map, tree_reduce
import jax.numpy as jnp
import optax
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
import jax.random as jr
from jax import jit
import gymnasium
import numpy as np
from functools import partial
from gpshield.agents import squash_sin
from gpshield.moments import Moments
from gpshield.common.stats import Summary_Stats
from gpshield.common.distributions import (
    Categorical,
    Distribution,
    Independent,
    MultivariateNormalDiag,
)

def summary_stats(values, prefix=None):
    stats = {
        'mean': np.mean(values),
        'std': np.std(values),
        'max': np.max(values),
        'min': np.min(values),
        'mag': np.max(np.abs(values)),
    }
    if prefix:
        stats = {f"{prefix}_{k}": v for k, v in stats.items() }
    return stats

@jit    
def compute_gradient_norm(grad):
    squared_norms = tree_map(lambda g: jnp.sum(jnp.square(g)), grad)
    total_squared_norm = tree_reduce(jnp.add, squared_norms, initializer=0.0)
    return jnp.sqrt(total_squared_norm)

def symlog(x):
  return np.sign(x) * np.log(1 + np.abs(x))

def symexp(x):
  return np.sign(x) * (np.exp(np.abs(x)) - 1)

@jit
def jsymlog(x):
  return jnp.sign(x) * jnp.log(1 + jnp.abs(x))

@jit
def jsymexp(x):
  return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)

def normalize_observation(obs, low, high, dim=0):
    finite_dims = np.isfinite(low) * np.isfinite(high)
    shape = finite_dims.shape + (1,) * dim
    low = low[finite_dims].reshape(shape)
    high = high[finite_dims].reshape(shape)
    normalized_obs = obs.copy()
    normalized_obs = np.moveaxis(normalized_obs, dim, 0)
    normalized_obs[finite_dims] = 2 * (normalized_obs[finite_dims] - low) / (high - low) -1
    return np.moveaxis(normalized_obs, 0, dim)

class ValueTrainState(TrainState):
    target_params: Any

class RolloutBuffer():

    def __init__(self, buffer_size, observation_space, action_space, n_envs, critics, device='auto', gae_lambda={'extr': 1., 'intr': 1.}, gamma={'extr': 0.99, 'intr': 0.99}):

        assert type(buffer_size) is int
        assert isinstance(observation_space, gymnasium.spaces.Box)
        assert isinstance(action_space, gymnasium.spaces.Box) or isinstance(action_space, gymnasium.spaces.Discrete)
        assert type(n_envs) is int
        assert type(device) is str
        assert type(n_envs) is int
        assert type(critics) is list
        assert len(critics) >= 1

        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_envs = n_envs
        self.critics = critics
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.pos = 0
        self.full = False

        if isinstance(self.action_space, gymnasium.spaces.Box):
            self.action_dim = int(np.prod(self.action_space.shape))
        elif isinstance(self.action_space, gymnasium.spaces.Discrete):
            self.action_dim = int(self.action_space.n)
        else:
            raise NotImplementedError

        self.input_shape = self.observation_space.shape

    def reset(self):

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.input_shape), dtype=np.float64)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float64)
        self.rewards = {k: np.zeros((self.buffer_size, self.n_envs), dtype=np.float64) for k in self.critics}
        self.returns = {k: np.zeros((self.buffer_size, self.n_envs), dtype=np.float64) for k in self.critics}
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float64)
        self.values = {k: np.zeros((self.buffer_size, self.n_envs), dtype=np.float64) for k in self.critics}
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float64)
        self.advantages = {k: np.zeros((self.buffer_size, self.n_envs), dtype=np.float64) for k in self.critics}
        self.weights = np.ones((self.buffer_size, self.n_envs), dtype=np.float64)
        self.pos = 0
        self.full = False

    def compute_returns_and_advantages(self, last_value=None, done=np.array(False)):

        if last_value is None:
            last_value = {k: np.array(0.0) for k in self.critics}

        for k in self.critics:

            last_gae_lam = 0

            for step in reversed(range(self.buffer_size)):
                if step == (self.buffer_size - 1):
                    next_non_terminal = 1.0 - done.astype(np.float64)
                    next_value = last_value[k]
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1].astype(np.float64)
                    next_value = self.values[k][step + 1]
                delta = self.rewards[k][step] + self.gamma[k] * next_value * next_non_terminal - self.values[k][step]
                last_gae_lam = delta + self.gamma[k] * self.gae_lambda[k] * next_non_terminal * last_gae_lam
                self.advantages[k][step] = last_gae_lam

            self.returns[k] = self.advantages[k] + self.values[k]

    def add(self, obs, action, reward, episode_start, value, log_prob, weight):

        if self.full:
            return False

        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.log_probs[self.pos] = np.array(log_prob)
        self.weights[self.pos] = np.array(weight)

        for k in self.critics:
            self.rewards[k][self.pos] = np.array(reward[k])
            self.values[k][self.pos] = np.array(value[k])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
        
        return True

    def get(self, key, batch_size):
        assert self.full
        indices = jr.permutation(key, self.buffer_size)

        if batch_size is None:
            batch_size=self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds):
        data = (self.observations[batch_inds].reshape(-1, *self.input_shape),
                self.actions[batch_inds].reshape(-1, self.action_dim),
                {k: self.rewards[k][batch_inds].flatten() for k in self.critics},
                {k: self.values[k][batch_inds].flatten() for k in self.critics},
                {k: self.returns[k][batch_inds].flatten() for k in self.critics},
                {k: self.advantages[k][batch_inds].flatten() for k in self.critics},
                self.log_probs[batch_inds].flatten(),
                self.weights[batch_inds].flatten())
        return data

class Flatten(nn.Module):
    """
    Equivalent to PyTorch nn.Flatten() layer.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape((x.shape[0], -1))

class Critic(nn.Module):
    n_units: int = 256
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(1)(x)
        return x


class Actor(nn.Module):
    action_dim: int
    n_units: int = 256
    log_std_init: float = 0.0
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    # For Discrete, MultiDiscrete and MultiBinary actions
    num_discrete_choices: Optional[Union[int, Sequence[int]]] = None
    # For MultiDiscrete
    max_num_choices: int = 0
    split_indices: np.ndarray = field(default_factory=lambda: np.array([]))


    def get_std(self) -> jnp.ndarray:
        # Make it work with gSDE
        return jnp.array(0.0)

    def __post_init__(self) -> None:
        # For MultiDiscrete
        if isinstance(self.num_discrete_choices, np.ndarray):
            self.max_num_choices = max(self.num_discrete_choices)
            # np.cumsum(...) gives the correct indices at which to split the flatten logits
            self.split_indices = np.cumsum(self.num_discrete_choices[:-1])
        super().__post_init__()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Distribution:
        x = Flatten()(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        action_logits = nn.Dense(self.action_dim)(x)
        if self.num_discrete_choices is None:
            # Continuous actions
            log_std = self.param("log_std", constant(self.log_std_init), (self.action_dim,))
            dist = MultivariateNormalDiag(loc=action_logits, scale_diag=jnp.exp(log_std))
        elif isinstance(self.num_discrete_choices, int):
            dist = Categorical(logits=action_logits)
        else:
            # Split action_logits = (batch_size, total_choices=sum(self.num_discrete_choices))
            action_logits = jnp.split(action_logits, self.split_indices, axis=1)
            # Pad to the maximum number of choices.
            # Pad by -inf, so that the probability of these invalid actions is 0.
            logits_padded = jnp.stack(
                [
                    jnp.pad(
                        logit,
                        # logit is of shape (batch_size, n)
                        # only pad after dim=1, to max_num_choices - n
                        # pad_width=((before_dim_0, after_0), (before_dim_1, after_1))
                        pad_width=((0, 0), (0, self.max_num_choices - logit.shape[1])),
                        constant_values=-np.inf,
                    )
                    for logit in action_logits
                ],
                axis=1,
            )
            dist = Independent(
                Categorical(logits=logits_padded), reinterpreted_batch_ndims=1
            )
        return dist


class A2C_Agent(JAXAgent):

    def __init__(
        self,
        key,
        observation_space,
        action_space,
        n_envs,
        n_steps=16,
        normalize_obs=False,
        normalize_adv=False,
        actor_kwargs={},
        actor_opt={},
        ent_coef=0.0,
        scales={'extr': 1.0, 'intr': 0.0},
        critic_kwargs={},
        device='auto',
    ):

        assert isinstance(key, jax._src.prng.PRNGKeyArray)
        assert isinstance(observation_space, gymnasium.spaces.Box)
        assert isinstance(action_space, gymnasium.spaces.Box) or isinstance(action_space, gymnasium.spaces.Discrete)
        assert type(n_envs) is int
        assert type(n_steps) is int
        assert type(normalize_obs) is bool
        assert type(normalize_adv) is bool
        assert isinstance(actor_kwargs, dict)
        assert isinstance(actor_opt, dict)
        assert type(ent_coef) is float
        assert isinstance(scales, dict) and ('extr' in scales) and ('intr' in scales)
        assert type(critic_kwargs) is dict
        assert (type(device) is str) and (device in ['cpu', 'gpu', 'auto'])

        self.observation_space = observation_space
        self.action_space = action_space
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.normalize_obs = normalize_obs
        self.normalize_adv = normalize_adv
        self.actor_kwargs = actor_kwargs
        self.actor_opt = actor_opt
        self.ent_coef = ent_coef
        self.critics = [k for k, v in scales.items() if v]
        self.scales = scales
        self.critic_kwargs = critic_kwargs
        self.device = device

        self.key = key

        self.norms = {k: Moments(**critic_kwargs[k].retnorm) for k in self.critics}
        self.gae_lambda = {k: critic_kwargs[k].gae_lambda for k in self.critics}
        self.gamma = {k: critic_kwargs[k].gamma for k in self.critics}
        self.polyak_tau = {k: critic_kwargs[k].polyak_tau for k in self.critics}
        self.slow_reg_coef = {k: critic_kwargs[k].slow_reg_coef for k in self.critics}

        if isinstance(self.action_space, gymnasium.spaces.Box):
            self.action_dim = int(np.prod(self.action_space.shape))
        elif isinstance(self.action_space, gymnasium.spaces.Discrete):
            self.action_dim = int(self.action_space.n)
            self.actor_kwargs.update({
                "num_discrete_choices": self.action_dim,
            })
        else:
            raise NotImplementedError

        self._setup_buffer()
        self._setup_optimizer()
        self._setup_model()

    def _setup_buffer(self):
        self.rollout_buffer = RolloutBuffer(
            self.n_steps, 
            self.observation_space, 
            self.action_space, 
            self.n_envs,
            self.critics,
            device=self.device, 
            gae_lambda=self.gae_lambda, 
            gamma=self.gamma
        )

    def _setup_optimizer(self):

        self.optimizer = {}

        if self.actor_opt.opt == 'adam':
            self.optimizer['actor'] = optax.adam
        elif self.actor_opt.opt == 'rms_prop':
            self.optimizer['actor'] = optax.rmsprop
        else:
            raise NotImplementedError

        for key in self.critics:
            if self.critic_kwargs[key].critic_opt.opt == 'adam':
                self.optimizer[f"{key}_critic"] = optax.adam
            elif self.critic_kwargs[key].critic_opt.opt == 'rms_prop':
                self.optimizer[f"{key}_critic"] = optax.rmsprop
            else:
                raise NotImplementedError

    def _setup_model(self):
        self.key, actor_key, vf_key = jr.split(self.key, 3)
        self.key, subkey = jr.split(self.key)

        self.reset_noise()

        obs = jnp.array([self.observation_space.sample()])

        self.actor = Actor(
            self.action_dim,
            #n_units=self.n_units,
            #log_std_init=self.log_std_init,
            #activation_fn=self.activation_fn,
            **self.actor_kwargs
        )

        self.actor.reset_noise = self.reset_noise

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(self.actor_opt.max_grad_norm),
                self.optimizer['actor'](
                    learning_rate=self.actor_opt.learning_rate, 
                    eps=self.actor_opt.eps
                ),
            ),
        )

        self.vf = {}
        self.vf_state = {}
        for key in self.critics:
            self.vf[key] = Critic(**self.critic_kwargs[key].critic)
            params = self.vf[key].init({"params": vf_key}, obs)
            self.vf_state[key] = ValueTrainState.create(
                apply_fn=self.vf[key].apply,
                params=params,
                tx=optax.chain(
                    optax.clip_by_global_norm(self.critic_kwargs[key].critic_opt.max_grad_norm),
                    self.optimizer[f"{key}_critic"](
                        learning_rate=self.critic_kwargs[key].critic_opt.learning_rate, 
                        eps=self.critic_kwargs[key].critic_opt.eps
                    ),
                ),
                target_params=params,
            )
            self.vf[key].apply = jit(self.vf[key].apply)

        self.actor.apply = jit(self.actor.apply)

    @staticmethod
    @jit
    def update_actor(actor_state, observations, actions, advantages, ent_coef, weights):

        def loss(params):
            dist = actor_state.apply_fn(params, observations)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()

            # Policy gradient loss
            policy_losses = -(advantages * log_prob)

            # Entropy loss favor exploration
            # Approximate entropy when no analytical form
            # entropy_loss = -jnp.mean(-log_prob)
            # analytical form
            entropy_losses = -entropy

            total_policy_losses = policy_losses + ent_coef * entropy_losses

            return (total_policy_losses * weights).mean()

        pg_loss, grads = jax.value_and_grad(loss, has_aux=False)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        return pg_loss, actor_state, compute_gradient_norm(grads)

    @staticmethod
    @jit
    def update_critic(vf_state, observations, returns, tau, weights, slow_reg_coef):

        target_vf_values = vf_state.apply_fn(vf_state.target_params, observations).flatten()

        def polyak_update(target_params, source_params, tau=0.02):
            return jax.tree_util.tree_map(lambda t, s: (1 - tau) * t + tau * s, target_params, source_params)

        def loss(params):
            vf_values = vf_state.apply_fn(params, observations).flatten()

            losses = (jsymlog(returns) - vf_values)**2
            reg = (vf_values - target_vf_values)**2
            losses += reg * slow_reg_coef
            return (losses * weights).mean()

        value_loss, grads = jax.value_and_grad(loss, has_aux=False)(vf_state.params)
        vf_state = vf_state.apply_gradients(grads=grads)

        new_target_params = polyak_update(vf_state.target_params, vf_state.params, tau)
        vf_state = vf_state.replace(target_params=new_target_params)
        return value_loss, vf_state, compute_gradient_norm(grads)

    def act(self, key, obs, deterministic=True):
        if self.normalize_obs:
            obs = normalize_observation(obs, self.observation_space.low, self.observation_space.high)
        if deterministic:
            action = self.select_action(self.actor_state, obs.reshape(1, -1))
        else:
            action = self.sample_action(key, self.actor_state, obs.reshape(1, -1))
        action = action.reshape(self.action_dim)
        clipped_action = jnp.clip(action, self.action_space.low, self.action_space.high)
        return clipped_action

    def propagate(self, obs, cov):
        raise NotImplementedError

    def predict_all(self, key, observation):
        return self._predict_all(key, self.actor_state, observation)

    def rollout(self, env):

        obs, info = env.reset()

        episode_start = True

        steps = 0
        self.rollout_buffer.reset()

        while steps < self.n_steps:
            self.reset_noise()

            obs_shape = obs.shape

            obs = obs.reshape(self.n_envs, -1)

            if self.normalize_obs:
                obs = normalize_observation(obs, self.observation_space.low, self.observation_space.high, dim=1)

            action, log_prob = self.predict_all(self.noise_key, obs)
            
            action = np.array(action)
            log_prob = np.array(log_prob)
            value = {k: np.array(self.vf_state[k].apply_fn(self.vf_state[k].params, obs).flatten()) for k in self.critics}
            value = {k: symexp(v) for k, v in value.items()}

            if "weight" in info.keys():
                weight = np.array(info["weight"]).flatten()
            else:
                weight = np.ones(len(obs))

            if isinstance(self.action_space, gymnasium.spaces.Box):
                clipped_action = np.clip(action, self.action_space.low, self.action_space.high)

            new_obs, reward, terminated, truncated, info = env.step(clipped_action)
            done = np.logical_or(terminated, truncated)

            reward = {k: np.array(reward[k]) for k in self.critics}

            steps += 1

            if isinstance(self.action_space, gymnasium.spaces.Discrete):
                # Reshape in case of discrete action
                action = action.reshape(-1, 1)

            #if done: #and (info['terminal']==False) and (info['truncated']==True):
            #    truncated_obs = new_obs
            #    truncated_value = np.array(
            #            self.vf.apply(  # type: ignore[union-attr]
            #                self.vf_state.params,
            #                truncated_obs.reshape(1, -1),
            #            ).flatten()
            #        ).item()
            #    reward += self.gamma * truncated_value

            obs = obs.reshape(obs_shape)

            self.rollout_buffer.add(
                obs,
                action,
                reward,
                episode_start,
                value,
                log_prob,
                weight
            )

            obs = new_obs
            episode_start = done
                
        assert isinstance(new_obs, np.ndarray) 

        new_obs = new_obs.reshape(self.n_envs, -1)

        if self.normalize_obs:
            new_obs = normalize_observation(new_obs, self.observation_space.low, self.observation_space.high, dim=1)

        last_value = {k: np.array(self.vf[k].apply(self.vf_state[k].params, new_obs).flatten()) for k in self.critics}
        last_value = {k: symexp(v) for k, v in last_value.items()}
        done = np.array(done)

        self.rollout_buffer.compute_returns_and_advantages(last_value=last_value, done=done)

        # compute returns and advantages
        return True

    def optimize(self, env):

        self.rollout(env)

        stats = {
            "weights": Summary_Stats(prefix="weights"),
            "actions": Summary_Stats(prefix="actions"),
            "actor_grad_norm": Summary_Stats(prefix="actor_grad_norm"),
            "actor_loss": Summary_Stats(prefix="actor_loss"),
        }

        for key in self.critics:
            stats.update({
                f"{key}_ret": Summary_Stats(prefix=f"{key}_ret"),
                f"{key}_val": Summary_Stats(prefix=f"{key}_val"),
                f"{key}_rew": Summary_Stats(prefix=f"{key}_rew"),
                f"{key}_adv": Summary_Stats(prefix=f"{key}_adv"),
                f'{key}_critic_grad_norm': Summary_Stats(prefix=f'{key}_critic_grad_norm'),
                f'{key}_critic_loss': Summary_Stats(prefix=f'{key}_critic_loss'),
            })

            if self.normalize_adv:
                stats.update({
                    f"{key}_normed_adv": Summary_Stats(prefix=f"{key}_normed_adv"),
                })
            else:
                stats.update({
                    f"{key}_normed_ret": Summary_Stats(prefix=f"{key}_normed_ret"),
                    f"{key}_normed_val": Summary_Stats(prefix=f"{key}_normed_val"),
                })

        self.key, subkey = jr.split(self.key)
        for rollout_data in self.rollout_buffer.get(subkey, None):

            observations, actions, rewards, values, returns, _, old_log_probs, weights = rollout_data

            advs = []
            total = sum(self.scales[k] for k in self.critics)
            for key in self.critics:
                ret = returns[key]
                rew = rewards[key]
                val = values[key]

                stats[f"{key}_ret"].update(ret)
                stats[f"{key}_val"].update(val)
                stats[f"{key}_rew"].update(rew)

                if self.normalize_adv:
                    adv = ret - val
                    offset, invscale = self.norms[key](adv)
                    normed_adv = (adv - offset) / invscale
                    advs.append(normed_adv * self.scales[key] / total)
                    stats[f"{key}_adv"].update(adv)
                    stats[f"{key}_normed_adv"].update(normed_adv)
                else:
                    offset, invscale = self.norms[key](ret)
                    normed_ret = (ret - offset) / invscale
                    normed_val = (val - offset) / invscale
                    adv = (normed_ret - normed_val) 
                    advs.append(adv * self.scales[key] / total)
                    stats[f"{key}_normed_ret"].update(normed_ret)
                    stats[f"{key}_normed_val"].update(normed_val)
                    stats[f"{key}_adv"].update(adv)

            advantages = np.stack(advs).sum(0)

            stats["actions"].update(actions)
            stats["weights"].update(weights)

            pg_loss, self.actor_state, actor_grad_norm = self.update_actor(self.actor_state, observations, actions, advantages, self.ent_coef, weights)

            stats['actor_grad_norm'].update(actor_grad_norm)
            stats['actor_loss'].update(pg_loss)

            for key in self.critics:
                value_loss, self.vf_state[key], vf_grad_norm = self.update_critic(self.vf_state[key], observations, returns[key], self.polyak_tau[key], weights, self.slow_reg_coef[key])

                stats[f'{key}_critic_grad_norm'].update(vf_grad_norm)
                stats[f'{key}_critic_loss'].update(value_loss)

        return stats

    def reset_noise(self):
        """
        Sample new weights for the exploration matrix, when using gSDE.
        """
        self.key, self.noise_key = jr.split(self.key)

    @staticmethod
    @jit
    def _predict_all(key, actor_state, observations):
        dist = actor_state.apply_fn(actor_state.params, observations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    @staticmethod
    @jit
    def select_action(actor_state, obs):
        return actor_state.apply_fn(actor_state.params, obs).mode()
        
    @staticmethod
    @jit
    def sample_action(key, actor_state, obs):
        dist = actor_state.apply_fn(actor_state.params, obs)
        action = dist.sample(seed=key)
        return action
