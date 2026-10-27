"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
import numpy as np
from envs.base import BaseEnv 
from agents import Linear_Agent
from scipy.spatial import ConvexHull

class ContinuousCartPole(BaseEnv):

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, reward_id, max_episode_steps=200, **kwargs):

        self.reward_id = reward_id

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 0.2095 # ~ 3/4 pi radians
        self.x_threshold = 2.4

        self.x_vel_threshold = 2.0
        self.theta_vel_threshold = 0.5

        min_action = np.array([-1.0])
        max_action = np.array([1.0])

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold*2,
                self.x_vel_threshold*2, #np.finfo(np.float32).max,
                self.theta_threshold_radians*2,
                self.theta_vel_threshold*2, #np.finfo(np.float32).max,
            ],
            dtype=np.float32
        )

        super().__init__((4,), (1,), -high, high, min_action, max_action, max_episode_steps=max_episode_steps, continuous=True, **kwargs)

        # rendering stuff
        self.render_mode = "rgb_array"
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

    def _step(self, state, action):

        x, x_dot, theta, theta_dot = state
        force = self.force_mag * float(action[0])
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        return np.array([x, x_dot, theta, theta_dot])

    def render(self):

        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return


        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )


        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

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

    @property
    def _init_state_function(self):
        return lambda: self.np_random.uniform(low=np.array([-0.05, -0.05, -0.05, -0.05]), high=np.array([0.05, 0.05, 0.05, 0.05]), size=(4,))

    @property
    def _reward_function(self):
        if self.reward_id == "rew_v1":
            return lambda s, a, ns: float(self._safe(ns))
        elif self.reward_id == "rew_v2":
            return lambda s, a, ns: float(ns[0] >= 0.1) * float(self._safe(ns)) 
        elif self.reward_id == "rew_v3":
            return lambda s, a, ns: float(ns[1] >= 0.1) * float(self._safe(ns)) 
        else:
            raise NotImplementedError

    @property 
    def _terminal_function(self):
        return lambda s: not self._safe(s)

    @property
    def _safety_constraints(self):
        return {
            'boundaries': [
                np.array([[-self.theta_threshold_radians, self.theta_threshold_radians]]),
            ],
        }
        
    @property
    def _safety_axes(self):
        return np.array([2])

    @property
    def _stable_set(self):
        return ConvexHull(self.hull_points())

    @property
    def _stable_axes(self):
        return np.array([1,2,3])

    @property
    def _backup(self):
        return Linear_Agent(None, 
                self.observation_space.shape[0], 
                self.action_space.shape[0], 
                action_limit=1.0, 
                weights=[[0.7488, 1.2280, 7.2758, 1.7787]],
            )

    @staticmethod
    def hull_points():
        return np.array(
            [[ 0.4875, -0.515 , -0.0118,  0.1203],
            [ 0.4304,  0.4783, -0.0806, -0.3115],
            [-0.3464, -0.6113,  0.1533, -0.0172],
            [-0.4035,  0.382 ,  0.0845, -0.4491],
            [ 0.4003, -0.3865, -0.1084,  0.4428],
            [ 0.3432,  0.6067, -0.1773,  0.011 ],
            [-0.4336, -0.4828,  0.0566,  0.3052],
            [-0.4907,  0.5104, -0.0122, -0.1266],
            [ 0.4907, -0.5104,  0.0122,  0.1266],
            [ 0.4336,  0.4828, -0.0566, -0.3052],
            [ 0.3432, -0.6067,  0.1773, -0.011 ],
            [-0.4003,  0.3865,  0.1084, -0.4428],
            [ 0.4035, -0.382 , -0.0845,  0.4491],
            [ 0.3464,  0.6113, -0.1533,  0.0172],
            [-0.4304, -0.4783,  0.0806,  0.3115],
            [-0.4875,  0.515 ,  0.0118, -0.1203]]
       )[:, 1:]