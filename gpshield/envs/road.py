from gpshield.envs.base import BaseEnv 
import numpy as np
from gpshield.agents import Linear_Agent

class ContinuousRoad(BaseEnv):

    metadata = {
        "render_modes": [],
        "render_fps": -1,
    }

    def __init__(self, reward_id, max_episode_steps=100, **kwargs):

        self.reward_id = reward_id

        self.dt = 1.0
        self.power = 0.0001
        self.max_speed = 0.1
        self.min_position = -4.0
        self.max_position = 4.0

        self.goal_position = 3.0

        min_action = np.array([-2.0])
        max_action = np.array([2.0])

        low = np.array(
            [
                self.min_position,
                -self.max_speed,
            ]
        )

        high = np.array(
            [
                self.max_position,
                self.max_speed,
            ]
        )

        super().__init__((2,), (1,), low, high, min_action, max_action,max_episode_steps=max_episode_steps, continuous=True, **kwargs) # obs_dim act_dim obs_low obs_high act_low act_high max_episode_steps= continuous=True **kwargs

        # render stuff
        self.render_mode = None


    def _step(self, state, action):
        state = state.copy()
        velocity = state[1] + 10.0 * action[0] * self.power
        position = state[0] + 10.0 * velocity * self.dt
        return np.array([position, velocity], dtype=np.float64)

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    @property
    def _init_state_function(self):
        return lambda: np.zeros(2)

    @property
    def _reward_function(self):
        if self.reward_id == "rew_v1":
            return lambda s, a, ns: ns[0] - s[0] if not self._done(ns) else 20.
        else:
            raise NotImplementedError

    @property
    def _terminal_function(self):
        return lambda s: s[0] >= self.goal_position

    @property
    def _safety_constraints(self):
        return {
            'boundaries': [
                np.array([[-0.01, 0.01]]),
            ],
        }

    @property
    def _safety_axes(self):
        return np.array([1])

    @property
    def _stable_set(self):
        return np.array([
            [-0.01, 0.01],
        ])

    @property
    def _stable_axes(self):
        return np.array([1])

    @property
    def _backup(self):
        return Linear_Agent(None,
                self.observation_space.shape[0],
                self.action_space.shape[0],
                action_limit=2.0,
                weights=[[0.0, -14.0425]]
        )
