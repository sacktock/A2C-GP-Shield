"""Smoke coverage for environment construction, model init, and training launch.

Imports of the training stack happen inside each test so a GPJAX migration
failure shows up as a test failure rather than a collection error.
"""

from pathlib import Path

import pytest

ENV_IDS = [
    "cartpole",
    "mountain_car",
    "obstacle",
    "obstacle2",
    "obstacle3",
    "obstacle4",
    "hopper",
    "road",
    "road_2d",
]


def _shrink_defaults(data):
    """Shrink the packaged defaults to a few steps and one optimisation update."""
    if not isinstance(data, dict) or "defaults" not in data:
        return data
    defaults = data["defaults"]
    defaults["run"].update(
        {
            "prefill": 4,
            "num_frames": 2,
            "post_hoc_frames": 2,
            "train_ratio": 2,
            "log_every": 1,
            "logger_window_size": 1,
        }
    )
    defaults["env"].update({"env_id": "cartpole", "max_episode_steps": 8, "obs_noise": 0.0})
    defaults["shield"]["lookahead"] = 1
    defaults["gp"].update(
        {
            "learning_rate": 0.01,
            "num_optim_iters": 1,
            "batch_size": 4,
            "buffer_size": 16,
            "sparse": False,
        }
    )
    defaults["ac"].update(
        {
            "num_posterior_samples": 1,
            "num_particles": 1,
            "rollout_steps": 2,
            "num_optim_iters": 1,
            "prefill_optim_iters": 1,
            "truncate_unsafe": False,
            "penalize_unsafe": False,
        }
    )
    defaults["ac"]["buffer"].update({"size": 16, "type": "default"})
    return data


@pytest.fixture
def tiny_defaults(monkeypatch):
    import ruamel.yaml as yaml

    real_load = yaml.YAML.load

    def load(self, stream):
        return _shrink_defaults(real_load(self, stream))

    monkeypatch.setattr(yaml.YAML, "load", load)


def _load_tiny_config():
    import ruamel.yaml as yaml

    from gpshield.common.config import Config

    config_path = Path(__file__).resolve().parents[1] / "gpshield" / "configs.yaml"
    raw = yaml.YAML(typ="safe").load(config_path.read_text(encoding="utf-8"))
    return Config(_shrink_defaults(raw)["defaults"])


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_make_env(env_id):
    from gpshield.run import make_env

    env = make_env(env_id, "rew_v1", max_episode_steps=5, obs_noise=0.0)
    obs, info = env.reset(seed=0)
    assert isinstance(info, dict)
    assert obs.shape == env.observation_space.shape


def test_init_dynamics_and_actor_critic(tiny_defaults):
    import jax.random as jr
    import numpy as np

    from gpshield.a2c import A2C_Agent
    from gpshield.common.buffer import TransitionBuffer
    from gpshield.dynamics_models import GP_Dynamics
    from gpshield.run import make_env

    config = _load_tiny_config()
    env = make_env(
        config.env.env_id,
        config.env.reward_id,
        max_episode_steps=config.env.max_episode_steps,
        obs_noise=config.env.obs_noise,
    )
    obs, _ = env.reset(seed=int(config.seed))

    states, actions, next_states = [], [], []
    for _ in range(int(config.run.prefill)):
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        next_obs, _, terminated, truncated, _ = env.step(action)
        states.append(np.asarray(obs))
        actions.append(np.asarray(action))
        next_states.append(np.asarray(next_obs))
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    buffer = TransitionBuffer(int(config.gp.buffer_size), env.observation_space, env.action_space)
    buffer.add(states, actions, next_states)

    key = jr.key(int(config.seed))
    key, subkey = jr.split(key)
    dynamics = GP_Dynamics(
        subkey,
        env,
        *buffer.get(),
        obs_noise=max(float(config.env.obs_noise), 1e-6),
        learning_rate=float(config.gp.learning_rate),
        num_iters=int(config.gp.num_optim_iters),
        batch_size=int(config.gp.batch_size),
        sparse=bool(config.gp.sparse),
    )
    assert dynamics.mgpr_models

    n_envs = int(config.ac.num_posterior_samples) * int(config.ac.num_particles)
    key, subkey = jr.split(key)
    agent = A2C_Agent(
        subkey,
        env.observation_space,
        env.action_space,
        n_envs,
        n_steps=int(config.ac.rollout_steps),
        normalize_obs=bool(config.ac.normalize_obs),
        normalize_adv=bool(config.ac.normalize_adv),
        actor_kwargs=config.ac.actor,
        actor_opt=config.ac.actor_opt,
        ent_coef=float(config.ac.ent_coef),
        scales=config.ac.critic_scales,
        critic_kwargs={"extr": config.ac.extr, "intr": config.ac.intr},
    )
    assert agent.n_envs == n_envs
    assert agent.action_dim >= 1


def test_training_launches(tiny_defaults):
    from gpshield.run import main

    main(["--tensorboard", "False", "--record_video", "False"])
