from gpshield.envs.base import BaseEnv 
import numpy as np
import math
import gymnasium
from gpshield.agents import Linear_Agent
from scipy.spatial import ConvexHull


class ContinuousMountainCar(BaseEnv):

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, reward_id, max_episode_steps=1000, **kwargs):

        self.reward_id = reward_id

        self.min_action = np.array([-1.0])
        self.max_action = np.array([1.0])
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = (
            0.45  # was 0.5 in gymnasium, 0.45 in Arnaud de Broissia's version
        )
        self.power = 0.0015
        self.gravity = 0.0025
        
        self.dt = 1.

        min_action = np.array([-1.0])
        max_action = np.array([1.0])

        high = np.array(
            [
                self.max_position,
                self.max_speed,
            ],
            dtype=np.float64
        )

        low = np.array(
            [
                self.min_position,
                -self.max_speed,
            ],
            dtype=np.float64
        )

        super().__init__((2,), (1,), low, high, min_action, max_action, max_episode_steps=max_episode_steps, continuous=True, **kwargs)

        # rendering stuff
        self.render_mode = "rgb_array"
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

    def _step(self, state, action):

        state = state.copy()
        velocity = state[1] + self.power * action[0] - self.gravity * np.cos(3 * state[0])
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position = state[0] + velocity * self.dt
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0
        return np.array([position, velocity], dtype=np.float64)

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                'e.g. by passing render_mode="rgb_array" to the environment constructor.'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise gymnasium.error.DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array":
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.max_position - self.min_position
        scale = self.screen_width / world_width
        carwidth = 40
        carheight = 20

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        pos = self.state[0]

        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )

        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + self._height(pos) * scale),
            )

            gfxdraw.aacircle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
            gfxdraw.filled_circle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )

        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        if self._render_override:
            gfxdraw.filled_circle(self.surf, 15, 375, 10, (255, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    @property
    def _init_state_function(self):
        return lambda: np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0.])

    @property
    def _reward_function(self):
        if self.reward_id == "rew_v1":
            return lambda s, a, ns: 100. if self._done(ns) else -0.1 * np.linalg.norm(a) ** 2
        elif self.reward_id == "rew_v2":
            def shaped_reward(s, a, ns):
                if self._done(ns):
                    return 100.
                else:
                    s_potential = self._height(s[0]) / 0.9 + (s[0] + (np.pi/6)) * s[1] / 0.07
                    ns_potential = self._height(ns[0]) / 0.9 + (ns[0] + (np.pi/6)) * ns[1] / 0.07
                    #acc_bonus = np.abs(0.99 * ns[1] - s[1])/0.004
                    return -0.1 * np.linalg.norm(a) ** 2 + 0.99 * ns_potential - s_potential #+ acc_bonus
            return shaped_reward
        elif self.reward_id == "rew_v3":
            def shaped_reward(s, a, ns):
                if self._done(ns):
                    return 100.
                else:
                    cost = 0.1 if np.sign(s[1]) == np.sign(a[0]) else 1.0
                    return -0.1 * np.linalg.norm(a) ** 2 - 0.1 * cost
            return shaped_reward
        else:
            raise NotImplementedError

    @property
    def _terminal_function(self):
        return lambda s: s[0] >= self.goal_position

    @property
    def _safety_constraints(self):
        return {
            'boundaries': [
                np.array([[self.min_position, np.inf]])
            ],
        }

    @property
    def _safety_axes(self):
        return np.array([0])

    @property
    def _stable_set(self):
        return ConvexHull(self.hull_points())

    @property
    def _stable_axes(self):
        return np.array([0,1])

    @property
    def _backup(self):
        return Linear_Agent(None, 
                self.observation_space.shape[0], 
                self.action_space.shape[0], 
                action_limit=1.0, 
                #weights=[[-0.0551, -11.4728]],
                weights=[[-0.7625, -34.5971]],
                input_bias=[-np.pi/6, 0.0],
            )

    @staticmethod
    def hull_points():
        return np.array([
            [0.6, 0.1],
            [-1.1, 0.1],
            [-1.1, -0.0226],
            [-np.pi/6, -0.062],
            [0.6, -0.0574],
        ])

    