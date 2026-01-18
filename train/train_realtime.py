"""
BeeWalker Training - Real-time Visualization
Uses shared_state for live MJPEG streaming to web dashboard.
"""

import os

# Set MuJoCo to use EGL for headless rendering
os.environ.setdefault('MUJOCO_GL', 'egl')

import threading
import time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from src.gym_env import BeeWalkerEnv
from src.flask_server import run_server
from src.shared_state import shared_state


class RealtimeVizCallback(BaseCallback):
    """
    Callback that updates shared_state with rendered frames for real-time visualization.
    """
    
    def __init__(self, render_env, render_freq=2, verbose=0):
        super().__init__(verbose)
        self.render_env = render_env
        self.render_freq = render_freq
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Track rewards
        rewards = self.locals.get("rewards", [0])
        self.current_episode_reward += rewards[0]
        
        # Render frame every N steps
        if self.n_calls % self.render_freq == 0:
            try:
                frame = self.render_env.render()
                if frame is not None:
                    shared_state.update_frame(frame)
            except Exception as e:
                pass
        
        # Check for episode end
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            
            # Update stats
            shared_state.update_stats(
                episode=self.episode_count,
                step=self.n_calls,
                reward=self.current_episode_reward,
                status="Training..."
            )
            
            if self.episode_count % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {self.episode_count}: Reward = {self.current_episode_reward:.2f}, Mean(100) = {mean_reward:.2f}")
            
            self.current_episode_reward = 0
        
        # Update step count periodically
        if self.n_calls % 100 == 0:
            shared_state.update_stats(step=self.n_calls)
        
        return True


def main():
    print("🐝 BeeWalker Training - Real-time Visualization")
    print("=" * 50)
    
    # Start Flask server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("✅ Web Dashboard started on http://localhost:1306")
    time.sleep(1)
    
    shared_state.update_stats(status="Creating environment...")
    
    # Create environments
    print("📦 Creating environments...")
    
    # Training env (no rendering needed for speed)
    train_env = DummyVecEnv([lambda: BeeWalkerEnv(render_mode=None)])
    
    # Render env (for visualization)
    render_env = BeeWalkerEnv(render_mode="rgb_array")
    
    shared_state.update_stats(status="Creating PPO agent...")
    
    # Create PPO agent
    print("🧠 Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        device="cpu",  # CPU is faster for MLP policies
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64])
        )
    )
    
    # Sync render env with training env initially
    obs, _ = render_env.reset()
    
    # Callback for real-time viz
    callback = RealtimeVizCallback(render_env, render_freq=2)
    
    shared_state.update_stats(status="Training...")
    
    print("🏋️ Starting training...")
    print("   Open http://localhost:1306 to watch!")
    print()
    
    try:
        model.learn(
            total_timesteps=10_000_000,
            callback=callback,
            progress_bar=False
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
    
    # Save model
    model.save("beewalker_model")
    print("💾 Model saved to beewalker_model.zip")
    
    shared_state.update_stats(status="Training complete!")
    
    # Cleanup
    train_env.close()
    render_env.close()


if __name__ == "__main__":
    main()
