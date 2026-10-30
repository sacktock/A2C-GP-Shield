from gpshield.envs.vec_env import VecEnv
import numpy as np
import jax.numpy as jnp


class GP_Posterior_Env(VecEnv):
    def __init__(
        self,
        key,
        num_posterior_samples,
        num_particles,
        env,
        dynamics_model,
        state_buffer,
        replay_beta=1.0,
        truncate_unsafe=False,
        penalize_unsafe=False,
        penalty_coef=1.0,
    ):

        assert type(num_posterior_samples) is int
        assert num_posterior_samples >= 1
        assert type(num_particles) is int
        assert num_particles >= 1

        self.num_posterior_samples = num_posterior_samples
        self.num_particles = num_particles

        self.env = env
        self.transition_fn = dynamics_model.sample_approx(
            key, num_samples=self.num_posterior_samples
        )

        self.state_buffer = state_buffer
        self.replay_beta = replay_beta
        self.truncate_unsafe = truncate_unsafe
        self.penalize_unsafe = penalize_unsafe
        self.penalty_coef = penalty_coef

        self.weight = None

        super().__init__(
            num_posterior_samples * num_particles,
            env.observation_space,
            env.action_space,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state, self.weight = self._init_state()
        return self.state, {"weight": self.weight}

    def step(self, action):
        action = action.astype(self.action_space.dtype)
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert np.all(
            self.action_space.contains(action[i]) for i in range(len(action))
        ), err_msg
        assert action.shape[0] == self.num_envs

        next_state = self._step(self.state, action)
        intr_rew = np.mean(np.std(next_state, axis=1), axis=1)

        next_state = next_state[:, 0, :]

        safe = self._safe(next_state)

        terminated = self._done(next_state)

        if self.truncate_unsafe:
            extr_rew = self._reward(self.state, action, next_state) * safe
            truncated = np.logical_not(safe)
        else:
            extr_rew = self._reward(self.state, action, next_state)
            truncated = np.zeros_like(terminated, dtype=bool)

        if self.penalize_unsafe:
            extr_rew += np.logical_not(safe).astype(np.float64) * -self.penalty_coef

        info = self._info(next_state)

        reward = {"extr": extr_rew, "intr": intr_rew}

        self.state = next_state

        done = np.logical_or(terminated, truncated)
        if np.any(done):
            inds = np.arange(self.num_envs)[done]
            self.state, self.weight = self._init_state(env_inds=inds)

        info.update({"weight": self.weight})

        return self.state, reward, terminated, truncated, info

    def _init_state(self, env_inds=None):
        if env_inds is None:
            batch_size = self.num_particles
            state_batch, weight_batch = self.state_buffer.sample(
                batch_size, beta=self.replay_beta
            )
            state = np.concatenate(
                [
                    np.tile(state.reshape(1, -1), (self.num_posterior_samples, 1))
                    for state in state_batch
                ]
            )
            weight = np.concatenate(
                [
                    np.tile(weight, (self.num_posterior_samples))
                    for weight in weight_batch
                ]
            )
            return state, weight
        else:
            state = self.state
            weight = self.weight
            batch_size = len(env_inds)
            state_batch, weight_batch = self.state_buffer.sample(
                batch_size, beta=self.replay_beta
            )
            state[env_inds] = state_batch
            weight[env_inds] = 0.0
            return state, weight

    def _step(self, state, action):
        assert state.shape[0] == self.num_envs
        x = jnp.concatenate([state, action], axis=-1)
        delta = self.transition_fn(x)
        return np.array(state.reshape(self.num_envs, 1, -1) + delta)

    def _reward(self, state, action, next_state):
        return np.array(
            [
                self._reward_function(state[i], action[i], next_state[i])
                for i in range(state.shape[0])
            ]
        )

    def _done(self, state):
        return np.array(
            [self._terminal_function(state[i]) for i in range(state.shape[0])]
        )

    @property
    def _init_state_function(self):
        return self.env.get_init_state_function()

    @property
    def _reward_function(self):
        return self.env.get_reward_function()

    @property
    def _terminal_function(self):
        return self.env.get_terminal_function()

    @property
    def _safety_constraints(self):
        return self.env.get_safety_constraints()

    @property
    def _safety_axes(self):
        return self.env.get_safety_axes()
