import numpy as np
import jax.numpy as jnp

from gpshield.envs.base import AbstractEnv

def is_inside(hull, x_new_ax):
    A = hull.equations[:, :-1] 
    b = hull.equations[:, -1]
    return np.all(x_new_ax @ A.T <= -b, axis=1)

def is_outside(hull, x_new_ax):
    A = hull.equations[:, :-1] 
    b = hull.equations[:, -1]
    return np.any(x_new_ax @ A.T > -b, axis=1)

class VecEnv(AbstractEnv):

    def __init__(self, num_envs, observation_space, action_space):

        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

        super().__init__()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._init_state()
        return self.state, {}

    def step(self, action):
        action = action.astype(self.action_space.dtype)
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert np.all(self.action_space.contains(action[i]) for i in range(len(action))), err_msg
        assert action.shape[0] == self.num_envs
        
        next_state = self._step(self.state, action)
        reward = self._reward(self.state, action, next_state)
        terminated = self._done(next_state)
        truncated = np.zeros_like(terminated, dtype=bool)
        info = self._info(next_state)

        self.state = next_state

        done = np.logical_or(terminated, truncated)
        if np.any(done):
            inds = np.arange(self.num_envs)[done]
            self.state = self._init_state(env_inds=inds)

        return self.state, reward, terminated, truncated, info

    def _safe(self, state):
        # only select the relevant state dimenions
        state_ax = np.array(state)[:, self._safety_axes]
        # check corresponding obstacle/boundary constraints
        for key, constraints in self._safety_constraints.items():
            for constraint in constraints:
                if isinstance(constraint, np.ndarray):
                    if key == "obstacles":
                        lower_bounds, upper_bounds = constraint[:,0], constraint[:,1]
                        return np.logical_not(np.all(state_ax > lower_bounds, axis=1) * np.all(state_ax < upper_bounds, axis=1))
                    elif key == "boundaries":
                        lower_bounds, upper_bounds = constraint[:,0], constraint[:,1]
                        return np.all(state_ax > lower_bounds, axis=1) * np.all(state_ax < upper_bounds, axis=1)
                    else:
                        raise NotImplementedError(f" Unexpected safety constraint type: {key}")
                elif isinstance(constraint, ConvexHull):
                    if key == "obstacles":
                        return is_outside(constraint, state_ax)
                    elif key == "boundaries":
                        return is_inside(constraint, state_ax)
                    else:
                        raise NotImplementedError(f" Unexpected safety constraint type: {key}")
                else:
                    raise NotImplementedError("Unexpected object for safety constraint")

class VecWrapper(VecEnv):

    def __init__(self, env, num_envs):
        self.env = env
        super().__init__(num_envs, env.observation_space, env.action_space)

        super().reset()

    def reset(self, *, seed=None, options=None):
        return self.state, {}

    def _reset(self):
        return super().reset()

    def _init_state(self, env_inds=None):
        if env_inds is None:
            return np.array([self._init_state_function() for i in range(self.num_envs)])
        else:
            state = self.state
            init_states = np.array([self._init_state_function() for _ in range(len(env_inds))])
            state[env_inds] = init_states
            return state

    def _step(self, state, action):
        return np.array([self.env.apply_step(state[i], action[i]) for i in range(state.shape[0])])

    def _reward(self, state, action, next_state):
        extr_rew = np.array([self._reward_function(state[i], action[i], next_state[i]) for i in range(state.shape[0])])
        return {'extr': extr_rew, 'intr': np.zeros(state.shape[0])}
    
    def _done(self, state):
        return np.array([self._terminal_function(state[i]) for i in range(state.shape[0])])

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




    

    








