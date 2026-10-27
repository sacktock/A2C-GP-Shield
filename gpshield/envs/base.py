import gymnasium
import numpy as np
from scipy.spatial import ConvexHull

def is_inside(hull, x_new_ax):
    A = hull.equations[:, :-1] 
    b = hull.equations[:, -1]
    return np.all(A @ x_new_ax <= -b)

def is_outside(hull, x_new_ax):
    A = hull.equations[:, :-1] 
    b = hull.equations[:, -1]
    return np.any(A @ x_new_ax + b > 0)

class AbstractEnv(gymnasium.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    render_mode = None

    def __init__(self):
        super().__init__()
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def get_reward_function(self):
        return self._reward_function

    def get_terminal_function(self):
        return self._terminal_function

    def get_init_state_function(self):
        return self._init_state_function

    def get_safety_constraints(self):
        return self._safety_constraints

    def get_safety_axes(self):
        return self._safety_axes

    def get_stable_set(self):
        return self._stable_set

    def get_stable_axes(self):
        return self._stable_axes

    def get_backup(self):
        return self._backup

    def _init_state(self, env_inds=None):
        raise NotImplementedError

    def _step(self, state, action):
        raise NotImplementedError

    def _reward(self, state, action, next_state):
        raise NotImplementedError

    def _safe(self, state):
        raise NotImplementedError

    def _done(self, state):
        raise NotImplementedError

    def _info(self, state):
        return {}

    @property
    def _init_state_function(self):
        raise NotImplementedError

    @property
    def _reward_function(self):
        raise NotImplementedError

    @property
    def _terminal_function(self):
        raise NotImplementedError

    @property
    def _safety_constraints(self):
        raise NotImplementedError

    @property
    def _safety_axes(self):
        raise NotImplementedError

    @property
    def _stable_set(self):
        raise NotImplementedError

    @property
    def _stable_axes(self):
        raise NotImplementedError

    @property
    def _backup(self):
        raise NotImplementedError

class BaseEnv(AbstractEnv):

    def __init__(self, obs_dim, act_dim, obs_low, obs_high, act_low, act_high, max_episode_steps=None, continuous=True, obs_noise=1e-6):

        assert obs_dim == obs_low.shape == obs_high.shape
        self.observation_space = gymnasium.spaces.Box(obs_low, obs_high, dtype=np.float64)

        self.obs_noise = obs_noise

        if continuous:
            assert act_dim == act_low.shape == act_high.shape
            self.action_space = gymnasium.spaces.Box(act_low, act_high, dtype=np.float64)
        else:
            assert len(act_dim.shape) == 1
            self.action_space = gymnasium.spaces.Discrete(act_dim)

        self._max_episode_steps = max_episode_steps

        self._elapsed_steps = 0

        self.steps_beyond_done = None

        self._render_override = False

        super().__init__()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._elapsed_steps = 0
        self.state = self._init_state()
        self.steps_beyond_done = None
        self._render_override = False
        return self.state, {}

    def step(self, action):
        action = action.astype(self.action_space.dtype)
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self._elapsed_steps += 1

        next_state = self._step(self.state, action)
        reward = self._reward(self.state, action, next_state)
        truncated = self._truncated()
        terminated = self._done(next_state)
        done = terminated or truncated
        info = self._info(next_state)
        info.update({"truncated": truncated})

        if done and (self.steps_beyond_done is None):
            self.steps_beyond_done = 0
        elif done and (self.steps_beyond_done is not None):
            if self.steps_beyond_done == 0:
                gymnasium.logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1

        self.state = next_state
        noise = self.np_random.standard_normal(self.observation_space.shape) * self.obs_noise
        return self.state + noise, reward, terminated, truncated, info

    def apply_step(self, state, action):
        return self._step(state, action)

    def set_render_override(self, val):
        self._render_override = val

    def _truncated(self):
        if self._max_episode_steps is not None:
            return self._elapsed_steps >= self._max_episode_steps
        else:
            return False

    def _init_state(self):
        return self._init_state_function()

    def _reward(self, state, action, next_state):
        return self._reward_function(state, action, next_state)

    def _safe(self, state):
        # only select the relevant state dimenions
        state_ax = np.array(state)[self._safety_axes]
        # check corresponding obstacle/boundary constraints
        for key, constraints in self._safety_constraints.items():
            for constraint in constraints:
                if isinstance(constraint, np.ndarray):
                    if key == "obstacles":
                        lower_bounds, upper_bounds = constraint[:,0], constraint[:,1]
                        return not (np.all(state_ax > lower_bounds) and np.all(state_ax < upper_bounds))
                    elif key == "boundaries":
                        lower_bounds, upper_bounds = constraint[:,0], constraint[:,1]
                        return np.all(state_ax > lower_bounds) and np.all(state_ax < upper_bounds)
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

    def _done(self, state):
        return self._terminal_function(state)

    def _info(self, state):
        return {'is_safe': self._safe(state)}

    @property
    def requires_reset(self):
        return (self.state is None) or (self.steps_beyond_done is not None)

    

    