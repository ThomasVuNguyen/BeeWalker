"""
BeeWalker Training Script - Stable Baselines3 + Gymnasium
Provides real-time visualization of training progress.
"""

import os
import threading
import time
import base64
import io

import numpy as np
import requests
import uvicorn
from PIL import Image

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

# Import our environment
from src.gym_env import BeeWalkerEnv
from src.web.server import app


class DashboardCallback(BaseCallback):
    """
    Callback to send training metrics and rendered frames to the web dashboard.
    """
    
    def __init__(self, eval_env, update_freq=1000, render_freq=5000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.update_freq = update_freq
        self.render_freq = render_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Track episode stats
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        
        # Send stats update
        if self.n_calls % self.update_freq == 0:
            mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
            try:
                requests.post("http://localhost:1306/update", json={
                    "generation": int(self.n_calls),
                    "mean_reward": float(mean_reward),
                    "status": f"Training... (Step {self.n_calls})"
                }, timeout=0.5)
            except:
                pass
            
            print(f"Step {self.n_calls}: Mean Reward = {mean_reward:.2f}")
        
        # Render and send frames for visualization
        if self.n_calls % self.render_freq == 0:
            self._send_viz_frames()
        
        return True
    
    def _send_viz_frames(self):
        """Render a short rollout and send frames to dashboard."""
        try:
            obs, _ = self.eval_env.reset()
            frames = []
            
            for _ in range(60):  # 60 frames = ~1.2 seconds at 50Hz
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                
                # Render to RGB array
                frame = self.eval_env.render()
                if frame is not None:
                    # Compress to JPEG and base64 encode
                    img = Image.fromarray(frame)
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=50)
                    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    frames.append(b64)
                
                if terminated or truncated:
                    break
            
            # Send frames to server
            if frames:
                requests.post("http://localhost:1306/viz", json={
                    "generation": int(self.n_calls),
                    "frames": frames
                }, timeout=5.0)
                
        except Exception as e:
            print(f"Viz error: {e}")


def run_server():
    """Start the web dashboard server."""
    uvicorn.run(app, host="0.0.0.0", port=1306, log_level="warning")


def main():
    print("🐝 BeeWalker Training - Gymnasium + SB3")
    print("=" * 50)
    
    # Start web server in background
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("✅ Web Dashboard started on port 1306")
    time.sleep(1)
    
    # Create environments
    print("📦 Creating environments...")
    train_env = DummyVecEnv([lambda: BeeWalkerEnv(render_mode=None)])
    eval_env = BeeWalkerEnv(render_mode="rgb_array")
    
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
        verbose=1,
        tensorboard_log="./logs/",
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64])  # Small network for Pico
        )
    )
    
    # Create callback
    callback = DashboardCallback(
        eval_env=eval_env,
        update_freq=2048,
        render_freq=10000
    )
    
    # Train
    print("🏋️ Starting training...")
    print("   Open http://localhost:1306 to watch!")
    print()
    
    try:
        model.learn(
            total_timesteps=10_000_000,
            callback=callback,
            progress_bar=False  # Disabled to avoid tqdm dependency
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
    
    # Save model
    model.save("beewalker_model")
    print("💾 Model saved to beewalker_model.zip")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    

if __name__ == "__main__":
    main()
