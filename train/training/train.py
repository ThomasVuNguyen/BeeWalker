"""Main training loop for BeeWalker."""

import os
import sys
import torch
import numpy as np
from gymnasium.wrappers import RecordVideo

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.environment import BeeWalkerEnv
from training.model import TransformerPolicy
from training.agent import PPOAgent
from common import shared_state


def train(max_episodes=10000, max_timesteps=1000, update_interval=2048):
    """
    Main training loop.
    
    Args:
        max_episodes: Maximum number of episodes to train
        max_timesteps: Maximum steps per episode
        update_interval: Update policy every N timesteps
    """
    print("=" * 60)
    print("BeeWalker Training")
    print("=" * 60)
    
    # Setup directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_dir = os.path.join(base_dir, "videos")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    success_path = os.path.join(checkpoint_dir, "beewalker_success.pth")
    
    # Create environment
    print("Creating environment...")
    try:
        env = BeeWalkerEnv(render_mode="rgb_array")
        
        # Wrap for video recording (every 100 episodes) - optional
        try:
            env = RecordVideo(
                env, 
                video_folder=video_dir,
                episode_trigger=lambda ep: ep % 100 == 0 or ep == 0,
                disable_logger=True
            )
            print("Environment created with video recording!")
        except Exception as video_err:
            print(f"Video recording disabled: {video_err}")
            print("Environment created without video recording.")
    except Exception as e:
        print(f"Error creating environment: {e}")
        raise
    
    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")
    
    # Create policy and agent
    policy = TransformerPolicy(obs_dim, act_dim, d_model=32, n_head=2, n_layers=2)
    agent = PPOAgent(policy)
    print(f"Policy parameters: {policy.get_param_count():,}")
    
    # Load checkpoint if exists
    start_episode = 1
    if agent.load_checkpoint(checkpoint_path):
        print("Resuming from checkpoint...")
    
    # Training metrics
    total_timesteps = 0
    best_reward = -float('inf')
    
    print("Starting training...")
    print("-" * 60)
    
    for episode in range(start_episode, max_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for t in range(max_timesteps):
            total_timesteps += 1
            
            # Select action
            action, logprob, value = agent.select_action(state)
            
            # Environment step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, logprob, reward, done, value)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            # Update frame for web UI (every 2 steps)
            if t % 2 == 0:
                try:
                    frame = env.render()
                    if frame is not None:
                        shared_state.update_frame(frame)
                except Exception:
                    pass  # Ignore render errors
            
            # PPO update
            if total_timesteps % update_interval == 0:
                loss = agent.update()
            
            if done:
                break
        
        # Update stats for web UI
        shared_state.update_stats(episode, episode_reward, episode_steps)
        
        # Track best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Logging
        if episode % 10 == 0:
            print(f"Episode {episode:5d} | Steps: {episode_steps:4d} | "
                  f"Reward: {episode_reward:8.2f} | Best: {best_reward:8.2f}")
        
        # Checkpointing
        if episode % 50 == 0:
            agent.save_checkpoint(checkpoint_path)
        
        # Success check (walked for many steps without falling)
        if episode_steps >= 800:
            print(f"\n{'='*60}")
            print(f"SUCCESS! Robot walked {episode_steps} steps!")
            print(f"{'='*60}\n")
            torch.save(policy.state_dict(), success_path)
            policy.export_for_pico(os.path.join(checkpoint_dir, "pico_weights.npz"))
    
    env.close()
    print("Training complete!")


if __name__ == "__main__":
    train()
