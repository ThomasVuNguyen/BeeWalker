"""
BeeWalker Training Entry Point
"""

import threading
import time
import requests
import uvicorn
from src.web.server import app
import os

# Disable XLA command buffers to prevent CUDA graph memory issues
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_command_buffer='

import jax
import jax.numpy as jnp
from src.algo import train, make_env_fn

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=1306, log_level="warning")

def progress_callback(num_steps, metrics):
    try:
        requests.post("http://localhost:1306/update", json={
            "generation": int(num_steps),
            "mean_reward": float(metrics.get("eval/episode_reward", 0.0)),
            "status": "Training..."
        }, timeout=0.1)
    except Exception as e:
        pass
    print(f"Step {num_steps}: Reward {metrics.get('eval/episode_reward', 0.0):.2f}")

# Visualization state
viz_env = None
jit_step = None
jit_inference = None
key = jax.random.PRNGKey(0)

def policy_params_fn(num_steps, make_policy, params):
    global viz_env, jit_step, jit_inference, key
    
    # Initialize env for viz if needed
    if viz_env is None:
        viz_env = make_env_fn()
        # Create reusable jitted step function
        def step_fn(state, action):
            return viz_env.step(state, action)
        jit_step = jax.jit(step_fn)

    # Create inference function for this policy
    inference_fn = make_policy(params)
    jit_inference = jax.jit(inference_fn)

    # Run a short rollout (e.g. 200 steps)
    viz_state = viz_env.reset(key) # Reset each time to see start behavior
    trajectory = []
    
    for _ in range(30):  # Short rollout for visualization
        key, subkey = jax.random.split(key)
        action, _ = jit_inference(viz_state.obs, subkey)
        viz_state = jit_step(viz_state, action)
        
        # Capture positions for visualization
        # pipeline_state.x.pos is (num_bodies, 3)
        # We need to convert to python list
        pos = viz_state.pipeline_state.x.pos
        trajectory.append(pos.tolist())
        
    # Send trajectory to server
    try:
        requests.post("http://localhost:1306/viz", json={
            "generation": int(num_steps),
            "trajectory": trajectory
        }, timeout=1.0)
    except Exception as e:
        pass

def main():
    print("🚀 Starting BeeWalker System...")
    
    # 1. Start Web Server in a daemon thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("✅ Web Server started on port 1306")
    
    # 2. Start Training
    print("🏋️ Starting Training on device:", jax.devices()[0])
    
    # Notify dashboard that we are compiling
    try:
        requests.post("http://localhost:1306/update", json={
            "generation": 0,
            "mean_reward": 0.0,
            "status": "Compiling JAX/CUDA Kernels..."
        }, timeout=1.0)
    except:
        pass
    
    train(
        progress_fn=progress_callback,
        policy_params_fn=policy_params_fn  # Re-enabled for visualization
    )
    
    # Keep main thread alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
