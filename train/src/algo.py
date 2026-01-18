"""
BeeWalker Training Algorithm (PPO)
"""

from typing import Tuple
import functools
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from src.sim import BeeWalkerEnv
import jax

def make_env_fn(config=None):
    return BeeWalkerEnv(config)

def train(
    num_timesteps: int = 100_000_000,
    num_evals: int = 10,
    reward_scaling: float = 10,
    episode_length: int = 200,
    normalize_observations: bool = True,
    action_repeat: int = 1,
    unroll_length: int = 5,
    num_minibatches: int = 8,
    num_updates_per_batch: int = 4,
    discounting: float = 0.97,
    learning_rate: float = 3e-4,
    entropy_cost: float = 1e-2,
    num_envs: int = 64,
    batch_size: int = 64,
    seed: int = 0,
    progress_fn = None,
    policy_params_fn = None
):
    # Create the environment
    # Note: We need to register it or use a wrapper if we want auto-vectorization
    # Brax's ppo.train takes an environment function
    
    env = make_env_fn()
    
    # We might need to wrap it for auto-reset and vectorization
    # Brax usually handles this inside ppo.train if we pass the right things
    # But for custom envs, we often need to be careful.
    
    # Defining a network definition (MLP)
    # Brax has default networks.
    
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        num_timesteps=num_timesteps,
        num_evals=num_evals,
        log_training_metrics=True,
        training_metrics_steps=100_000,
        normalize_observations=normalize_observations,
        episode_length=episode_length,
        action_repeat=action_repeat,
        unroll_length=unroll_length,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
        num_updates_per_batch=num_updates_per_batch,
        num_envs=num_envs,
        learning_rate=learning_rate,
        entropy_cost=entropy_cost,
        discounting=discounting,
        seed=seed,
        progress_fn=progress_fn,
        num_eval_envs=8,  # Reduce eval memory usage
        run_evals=False,  # Skip evaluation to avoid OOM
        # Only pass policy_params_fn if it's not None
    )
    
    return make_inference_fn, params
