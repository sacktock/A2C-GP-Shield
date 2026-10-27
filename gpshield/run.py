from jax import config
config.update("jax_enable_x64", True)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

from gymnasium.wrappers import RecordVideo

import argparse
import importlib
import jax.random as jr
import jax.numpy as jnp
import time
import math

import numpy as np
from scipy.special import softmax

from gpshield.dynamics_models import GP_Dynamics

from gpshield.agents import Random_Agent, Linear_Agent

from gpshield.common.logger import RolloutLogger, TrainLogger
from gpshield.common.buffer import TransitionBuffer, StateBuffer, PrioritizedStateBuffer, NoveltyStateBuffer
from gpshield.common.config import Config, Flags, Path

import tensorflow as tf

from gpshield.a2c import A2C_Agent

from gpshield.envs.gp_posterior_env import GP_Posterior_Env

from tqdm import tqdm

from gpshield.envs.vec_env import VecWrapper

import ruamel.yaml as yaml

def make_env(env_id, reward_id, max_episode_steps=200, **kwargs):
    ctor = {
        'cartpole': 'envs.cartpole:ContinuousCartPole',
        'mountain_car': 'envs.mountain_car:ContinuousMountainCar',
        'obstacle': 'envs.obstacle:ContinuousObstacle',
        'obstacle2': 'envs.obstacle2:ContinuousObstacle2',
        'obstacle3': 'envs.obstacle3:ContinuousObstacle3',
        'obstacle4': 'envs.obstacle4:ContinuousObstacle4',
        'hopper': 'envs.hopper:HopperV5',
        'road': 'envs.road:ContinuousRoad',
        'road_2d': 'envs.road_2d:ContinuousRoad_2d',
    }[env_id]
    if isinstance(ctor, str):
        module, cls = ctor.split(':')
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    env = ctor(reward_id, max_episode_steps=max_episode_steps, **kwargs)
    return env

def make_replay_buffer(args, env, impl="default"):
    if impl == "default":
        return StateBuffer(args.size, env.observation_space, env.action_space)
    elif impl == "return":
        return PrioritizedStateBuffer(args.size, env.observation_space, env.action_space, alpha=args.alpha)
    elif impl == "novelty":
        return NoveltyStateBuffer(args.size, env.observation_space, env.action_space, alpha=args.alpha, n_bins=args.n_bins)
    else:
        raise NotImplementedError

def rollout(key, env, agent, timesteps, logger=None):

    s, a, r, d, ns = [], [], [], [], []

    if env.requires_reset:
        state, _ = env.reset()
        if logger is not None:
            logger.reset()
    else:
        state = env.state

    for i in tqdm(range(timesteps)):
        key, subkey = jr.split(key)
        action = agent.policy(subkey, state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if logger is not None:
            logger.step({
                    'done' : done, 
                    'ep_rew' : reward, 
                    'ep_len' : 1.0, 
                    'ep_overrides' : 0.0, 
                    'is_success' : float(info['is_safe'])
                })

        s.append(state)
        a.append(action)
        r.append(reward)
        d.append(done)
        ns.append(next_state)
    
        state = next_state

        if done:
            state, _ = env.reset()

    return s, a, r, d, ns

def rollout_with_shield(key, env, agent, dynamics_model, shield_fn, timesteps, logger=None):

    s, a, r, d, ns = [], [], [], [], []

    if env.requires_reset:
        state, _ = env.reset()
        if logger is not None:
            logger.reset()
    else:
        state = env.state

    for i in tqdm(range(timesteps)):
        key, subkey = jr.split(key)
        action, override = shield_fn(subkey, state, agent, dynamics_model)
        env.set_render_override(override)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if logger is not None:
            logger.step({
                    'done' : done, 
                    'ep_rew' : reward, 
                    'ep_len' : 1.0, 
                    'ep_overrides' : float(override), 
                    'is_success' : float(info['is_safe'])
                })

        s.append(state)
        a.append(action)
        r.append(reward)
        d.append(done)
        ns.append(next_state)

        state = next_state

        if done:
            state, _ = env.reset()

    return s, a, r, d, ns

def main(argv=None):

    parsed, other = Flags(configs=['defaults']).parse_known(argv)
    configs = yaml.YAML(typ='safe').load(
      (Path(__file__).parent / 'configs.yaml').read())
    config = Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = Flags(config).parse(other)

    print(config)

    seed = int(config.seed)
    tensorboard = config.tensorboard
    record_video = config.record_video

    # Make environment and set the random seed
    env = make_env(config.env.env_id, config.env.reward_id, max_episode_steps=config.env.max_episode_steps, obs_noise=config.env.obs_noise)
    if record_video:
        env = RecordVideo(env, video_folder=f"./videos/{config.env.env_id}/{config.env.reward_id}", episode_trigger=lambda x: True, name_prefix="shield")
    key = jr.key(seed)
    env.reset(seed=seed)

    eval_env = make_env(config.env.env_id, config.env.reward_id, max_episode_steps=config.env.max_episode_steps, obs_noise=config.env.obs_noise)
    if record_video:
        eval_env = RecordVideo(eval_env, video_folder=f"./videos/{config.env.env_id}/{config.env.reward_id}", step_trigger=lambda x: (x % config.run.post_hoc_frames) == 0, name_prefix="eval")
    eval_env.reset(seed=seed)

    # Setup tensorboard logging
    if tensorboard:
        summary_writer = tf.summary.create_file_writer(config.logdir)
    else:
        summary_writer = None

    # Collect initial data for the GP dynamics model with the safe backup policy
    print("Collecting initial data with the safe backup policy ...")
    start_time = time.time()

    transition_buffer = TransitionBuffer(config.gp.buffer_size, env.observation_space, env.action_space)
    state_buffer = make_replay_buffer(config.ac.buffer, env, impl=config.ac.buffer.type)

    rollout_logger = RolloutLogger(config.run.log_every, tensorboard=tensorboard, tqdm=True, summary_writer=summary_writer, stats_window_size=config.run.logger_window_size, prefix='rollout/prefill')

    key, subkey = jr.split(key)
    if config.run.prefill_policy == "backup":
        prefill_agent = env.get_backup()
    elif config.run.prefill_policy == "random":
        prefill_agent = Random_Agent(env.action_space, continuous=True)
    else:
        raise NotImplementedError

    key, subkey = jr.split(key)
    s_new, a_new, r_new, d_new, ns_new = rollout(subkey, env, prefill_agent, config.run.prefill, logger=rollout_logger)

    transition_buffer.add(s_new, a_new, ns_new)
    state_buffer.add(s_new)

    end_time = time.time()
    print(f"Completed in {end_time - start_time} seconds")

    train_logger = TrainLogger(1, tqdm=True, tensorboard=tensorboard, summary_writer=summary_writer, stats_window_size=1, prefix='train')

    # Setup the GP dynamics model
    print(f"Optimizing the GP dynamics model ...")
    start_time = time.time()

    stats = {}

    # Use sparse GPR if specified
    key, subkey = jr.split(key)
    gp_args = (subkey, env, *transition_buffer.get())
    gp_kwargs = {"obs_noise" : config.env.obs_noise, "learning_rate" : config.gp.learning_rate, "num_iters" : config.gp.num_optim_iters, "batch_size" : config.gp.batch_size}
    if config.gp.sparse:
        print(f"Using sparse GPR with {'collapsed' if config.gp.collapsed else 'uncollapsed'} objective ...")
        gp_kwargs.update({"sparse": True, "collapsed": config.gp.collapsed, "num_inducing_inputs": config.gp.num_inducing_inputs})
        dynamics_model = GP_Dynamics(*gp_args, **gp_kwargs)
    else:
        dynamics_model = GP_Dynamics(*gp_args, **gp_kwargs)

    key, subkey = jr.split(key)
    stats.update(dynamics_model.optimize(subkey))

    end_time = time.time()
    print(f"Completed in {end_time - start_time} seconds")

    # Setup the neural actor-critic policy 
    print(f"Optimizing the actor critic policy  ...")
    start_time = time.time()

    beta_by_frame = lambda frame_idx: min(1.0, config.ac.buffer.beta_start + frame_idx * (1.0 - config.ac.buffer.beta_start) / config.ac.buffer.beta_frames)

    key, subkey = jr.split(key)
    stats["beta"] = config.ac.buffer.beta_start
    vec_posterior_env = GP_Posterior_Env(subkey, config.ac.num_posterior_samples, config.ac.num_particles, env, dynamics_model, state_buffer, replay_beta=config.ac.buffer.beta_start, truncate_unsafe=config.ac.truncate_unsafe, penalize_unsafe=config.ac.penalize_unsafe, penalty_coef=config.ac.penalty_coef)

    key, subkey = jr.split(key)
    a2c_kwargs = {"n_steps": config.ac.rollout_steps, "normalize_adv": config.ac.normalize_adv, "normalize_obs": config.ac.normalize_obs, "actor_kwargs": config.ac.actor, "actor_opt": config.ac.actor_opt, "ent_coef": config.ac.ent_coef, "scales": config.ac.critic_scales, "critic_kwargs": {"extr": config.ac.extr, "intr": config.ac.intr}}
    a2c_agent = A2C_Agent(subkey, vec_posterior_env.observation_space, vec_posterior_env.action_space, config.ac.num_posterior_samples*config.ac.num_particles, **a2c_kwargs)

    for _ in range(config.ac.prefill_optim_iters):    
        summary_stats = a2c_agent.optimize(vec_posterior_env)
        stats.update({k: v + stats[k] if k in stats else v for k, v in summary_stats.items()})

    end_time = time.time()
    print(f"Completed in {end_time - start_time} seconds")

    train_logger.step(stats)

    '''rollout_logger = RolloutLogger(config.run.log_every, tqdm=True, stats_window_size=config.run.logger_window_size, prefix='rollout/a2c/')
    
    env.reset()
    vec_env = VecWrapper(env, config.ac.num_particles)

    key, subkey = jr.split(key)
    a2c_agent = A2C_Agent(subkey, vec_env.observation_space, vec_env.action_space, config.ac.num_particles, **a2c_kwargs)

    eval_env.reset()

    for iters in range(math.ceil(config.run.num_frames/config.run.train_ratio)):
        key, subkey = jr.split(key)
        s_new, a_new, r_new, d_new, ns_new = rollout(subkey, eval_env, a2c_agent, config.run.train_ratio, logger=rollout_logger)

        stats = {}

        vec_env._reset()

        for _ in range(config.ac.num_optim_iters):
            summary_stats = a2c_agent.optimize(vec_env)
            stats.update({k: v + stats[k] if k in stats else v for k, v in summary_stats.items()})

        train_logger.step(stats)

    assert False'''

    '''rollout_logger = RolloutLogger(config.run.log_every, tqdm=True, stats_window_size=config.run.logger_window_size, prefix='rollout/a2c/')
    env.reset()

    for iters in range(math.ceil(config.run.num_frames/config.run.train_ratio)):
        key, subkey = jr.split(key)
        s_new, a_new, r_new, d_new, ns_new = rollout(subkey, env, a2c_agent, config.run.train_ratio, logger=rollout_logger)
        transition_buffer.add(s_new, a_new, ns_new)
        state_buffer.add(s_new)

        dynamics_model.set_experience(*transition_buffer.get())

        stats = {}

        key, subkey = jr.split(key)
        stats.update(dynamics_model.optimize(subkey))

        key, subkey = jr.split(key)
        stats["beta"]=beta_by_frame(iters*config.run.train_ratio)
        vec_posterior_env = GP_Posterior_Env(subkey, config.ac.num_posterior_samples, config.ac.num_particles, env, dynamics_model, state_buffer, replay_beta=stats["beta"], truncate_unsafe=config.ac.truncate_unsafe)

        for _ in range(config.ac.num_optim_iters):
            summary_stats = a2c_agent.optimize(vec_posterior_env)
            stats.update({k: v + stats[k] if k in stats else v for k, v in summary_stats.items()})
        
        train_logger.step(stats)
    
    assert False'''

    print(f"Commencing training {'with' if tensorboard else 'without'} logging ...")
    start_time = time.time()

    rollout_logger = RolloutLogger(config.run.log_every, tqdm=True, tensorboard=tensorboard, summary_writer=summary_writer, stats_window_size=config.run.logger_window_size, prefix='rollout/shield/')
    eval_logger = RolloutLogger(config.run.post_hoc_frames, tqdm=True, tensorboard=tensorboard, summary_writer=summary_writer, stats_window_size=20, prefix='rollout/eval/')

    # Construct the shield function
    safety_constraints = env.get_safety_constraints()
    safety_axes = env.get_safety_axes()
    stable_set = env.get_stable_set()
    stable_axes = env.get_stable_axes()
    safe_agent = env.get_backup()
    shield_fn = lambda key, state, agent, dynamics_model: dynamics_model.shield(subkey, state, agent, safe_agent, config.shield.lookahead, safety_constraints, safety_axes, stable_set, stable_axes, sigma_coef=config.shield.sigma_coef, method='gaussian_approx')
    
    # Reset the environments
    env.reset()

    # Start training
    for iters in range(math.ceil(config.run.num_frames/config.run.train_ratio)):
        # Rollout with the shield
        key, subkey = jr.split(key)
        s_new, a_new, r_new, d_new, ns_new = rollout_with_shield(subkey, env, a2c_agent, dynamics_model, shield_fn, config.run.train_ratio, logger=rollout_logger)
        transition_buffer.add(s_new, a_new, ns_new)
        state_buffer.add(s_new)

        dynamics_model.set_experience(*transition_buffer.get())

        stats = {}

        # Optimize the GP dynamics model
        key, subkey = jr.split(key)
        stats.update(dynamics_model.optimize(subkey))

        # Sample transition functions from the GP posterior
        key, subkey = jr.split(key)
        stats["beta"]=beta_by_frame(iters*config.run.train_ratio)
        vec_posterior_env = GP_Posterior_Env(subkey, config.ac.num_posterior_samples, config.ac.num_particles, env, dynamics_model, state_buffer, replay_beta=stats["beta"], truncate_unsafe=config.ac.truncate_unsafe, penalize_unsafe=config.ac.penalize_unsafe, penalty_coef=config.ac.penalty_coef)

        # Optimize the a2c policy
        for _ in range(config.ac.num_optim_iters):
            summary_stats = a2c_agent.optimize(vec_posterior_env)
            stats.update({k: v + stats[k] if k in stats else v for k, v in summary_stats.items()})
        
        train_logger.step(stats)

        key, subkey = jr.split(key)
        eval_env.reset()
        rollout(subkey, eval_env, a2c_agent, config.run.post_hoc_frames, logger=eval_logger)

    end_time = time.time()
    print(f"Completed in {end_time - start_time} seconds")

    # Evaluate the a2c agent without the shield fn
    print(f"Commencing post-hoc evaluation without shield ...")
    start_time = time.time()
    # Reset the environment
    eval_env.reset()

    rollout_logger = RolloutLogger(config.run.log_every, tensorboard=tensorboard, tqdm=True, summary_writer=summary_writer, stats_window_size=5, prefix='rollout/post_hoc/')

    # Collect eval rollouts
    key, subkey = jr.split(key)
    rollout(subkey, eval_env, a2c_agent, config.run.post_hoc_frames, logger=rollout_logger)

    end_time = time.time()
    print(f"Completed in {end_time - start_time} seconds")

if __name__ == "__main__":
    main()





