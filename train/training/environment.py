"""BeeWalker Custom Gym Environment."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os


class BeeWalkerEnv(gym.Env):
    """
    Custom Gym environment for BeeWalker bipedal robot.
    
    Observation Space (18D):
        - 6 joint positions (normalized)
        - 6 joint velocities (normalized)
        - 6 IMU values (3 accelerometer + 3 gyroscope)
    
    Action Space (6D):
        - 6 servo position commands (normalized -1 to 1)
        
    Reward:
        - Forward velocity reward
        - Upright bonus
        - Energy penalty
        - Fall penalty
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        
        # Load MuJoCo model
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "beewalker.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Renderer for visualization
        self._renderer = None
        self._render_width = 640
        self._render_height = 480
        
        # Action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        
        # Episode parameters
        self.max_episode_steps = 1000
        self._step_count = 0
        
        # Joint ranges for denormalization (from XML)
        self.joint_ranges = np.array([
            [-60, 60],   # left_hip
            [-90, 0],    # left_knee
            [-30, 45],   # left_ankle
            [-60, 60],   # right_hip
            [-90, 0],    # right_knee
            [-30, 45],   # right_ankle
        ], dtype=np.float32)
        
        # Initial position
        self._init_qpos = None
        self._init_qvel = None

    def _get_obs(self):
        """Get observation from current state."""
        # Joint positions (6): normalized to [-1, 1]
        joint_pos = self.data.sensordata[:6].copy()  # Joint position sensors
        joint_pos_norm = np.zeros(6, dtype=np.float32)
        for i in range(6):
            low, high = self.joint_ranges[i]
            joint_pos_norm[i] = 2.0 * (joint_pos[i] - low) / (high - low) - 1.0
        
        # Joint velocities (6): scaled
        joint_vel = self.data.sensordata[6:12].copy()  # Joint velocity sensors
        joint_vel_norm = np.clip(joint_vel / 10.0, -1.0, 1.0).astype(np.float32)
        
        # IMU data (6): accelerometer (3) + gyroscope (3)
        accel = self.data.sensordata[12:15].copy()  # Accelerometer
        gyro = self.data.sensordata[15:18].copy()   # Gyroscope
        imu_data = np.concatenate([accel / 20.0, gyro / 10.0]).astype(np.float32)
        
        return np.concatenate([joint_pos_norm, joint_vel_norm, imu_data])

    def _get_reward(self):
        """Calculate reward based on current state."""
        # Torso position and velocity from sensors
        torso_pos = self.data.sensordata[18:21]  # framepos
        torso_vel = self.data.sensordata[25:28]  # framelinvel
        
        # Forward velocity reward (moving in +x direction)
        forward_vel = torso_vel[0]
        forward_reward = forward_vel * 2.0
        
        # Upright bonus (torso height)
        height = torso_pos[2]
        upright_reward = 0.5 if height > 0.25 else -0.5
        
        # Energy penalty (punish large actions)
        energy_penalty = -0.01 * np.sum(np.square(self.data.ctrl))
        
        # Alive bonus
        alive_bonus = 0.1
        
        return forward_reward + upright_reward + energy_penalty + alive_bonus

    def _is_terminated(self):
        """Check if episode should terminate (robot fell)."""
        # Get torso height
        torso_pos = self.data.sensordata[18:21]
        height = torso_pos[2]
        
        # Terminate if torso too low (fallen)
        if height < 0.15:
            return True
        
        # Check if torso is tilted too much (from quaternion)
        torso_quat = self.data.sensordata[21:25]
        # Approximate upright check using z-component
        # quat = [w, x, y, z], upright when close to [1, 0, 0, 0]
        if abs(torso_quat[0]) < 0.5:  # Tilted more than ~60 degrees
            return True
        
        return False

    def _is_truncated(self):
        """Check if episode should be truncated (max steps)."""
        return self._step_count >= self.max_episode_steps

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Add small random perturbation to initial state
        if self.np_random is not None:
            noise = self.np_random.uniform(-0.01, 0.01, size=self.model.nq)
            self.data.qpos[:] += noise
        
        # Step once to settle
        mujoco.mj_forward(self.model, self.data)
        
        self._step_count = 0
        
        obs = self._get_obs()
        info = {}
        
        return obs, info

    def step(self, action):
        """Take a step in the environment."""
        # Convert normalized action to joint angles
        action = np.clip(action, -1.0, 1.0)
        
        for i in range(6):
            low, high = self.joint_ranges[i]
            self.data.ctrl[i] = low + (action[i] + 1.0) * (high - low) / 2.0
        
        # Simulate multiple substeps for stability
        n_substeps = 4
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)
        
        self._step_count += 1
        
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = {"step": self._step_count}
        
        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self._render_height, self._render_width)
        
        self._renderer.update_scene(self.data, camera="track")
        pixels = self._renderer.render()
        
        return pixels

    def close(self):
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# Register the environment
gym.register(
    id="BeeWalker-v1",
    entry_point="training.environment:BeeWalkerEnv",
    max_episode_steps=1000,
)
