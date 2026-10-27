"""Gymnasium contract checks for the custom environments.

Env classes are imported directly so these tests do not pull in the GP
dynamics stack.
"""

import numpy as np
import pytest

import gymnasium
from gpshield.envs.cartpole import ContinuousCartPole
from gpshield.envs.mountain_car import ContinuousMountainCar
from gpshield.envs.obstacle import ContinuousObstacle
from gpshield.envs.obstacle2 import ContinuousObstacle2
from gpshield.envs.obstacle3 import ContinuousObstacle3
from gpshield.envs.obstacle4 import ContinuousObstacle4
from gpshield.envs.road import ContinuousRoad
from gpshield.envs.road_2d import ContinuousRoad_2d

ENV_CLASSES = [
    ContinuousCartPole,
    ContinuousMountainCar,
    ContinuousObstacle,
    ContinuousObstacle2,
    ContinuousObstacle3,
    ContinuousObstacle4,
    ContinuousRoad,
    ContinuousRoad_2d,
]


def _make(cls, seed=None):
    env = cls("rew_v1", max_episode_steps=20, obs_noise=0.0)
    obs, info = env.reset(seed=seed)
    return env, obs, info


@pytest.mark.parametrize("cls", ENV_CLASSES, ids=lambda cls: cls.__name__)
def test_spaces_are_gymnasium_spaces(cls):
    env, _, _ = _make(cls, seed=0)
    assert isinstance(env.observation_space, gymnasium.spaces.Space)
    assert isinstance(env.action_space, gymnasium.spaces.Space)
    assert isinstance(env.observation_space, gymnasium.spaces.Box)
    assert isinstance(env.action_space, gymnasium.spaces.Box)


@pytest.mark.parametrize("cls", ENV_CLASSES, ids=lambda cls: cls.__name__)
def test_reset_returns_obs_and_info(cls):
    env, obs, info = _make(cls, seed=0)
    assert isinstance(info, dict)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize("cls", ENV_CLASSES, ids=lambda cls: cls.__name__)
def test_reset_seed_is_reproducible(cls):
    env_a, obs_a, _ = _make(cls, seed=123)
    env_b, obs_b, _ = _make(cls, seed=123)
    np.testing.assert_allclose(obs_a, obs_b)
    np.testing.assert_allclose(env_a.np_random.random(5), env_b.np_random.random(5))


@pytest.mark.parametrize("cls", ENV_CLASSES, ids=lambda cls: cls.__name__)
def test_step_returns_gymnasium_tuple(cls):
    env, obs, _ = _make(cls, seed=0)
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    assert env.action_space.contains(action)

    result = env.step(action)
    assert len(result) == 5
    next_obs, reward, terminated, truncated, info = result

    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == env.observation_space.shape
    assert env.observation_space.contains(next_obs)
    assert np.ndim(reward) == 0
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert isinstance(info, dict)
    assert "is_safe" in info
    assert obs.shape == env.observation_space.shape
