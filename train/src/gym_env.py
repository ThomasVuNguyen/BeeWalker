"""
BeeWalker Environment - Gymnasium + MuJoCo
A bipedal walking robot environment compatible with Stable Baselines3.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

from src.config import ROBOT_CONFIG


class BeeWalkerEnv(gym.Env):
    """
    BeeWalker Gymnasium Environment using MuJoCo for physics.
    
    Observation Space: Joint positions, velocities, and IMU-like readings
    Action Space: 6 motor torques (3 per leg)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.config = ROBOT_CONFIG
        
        # Create MuJoCo model from MJCF
        self.mjcf_path = self._create_mjcf_file()
        self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.data = mujoco.MjData(self.model)
        
        # Action space: 6 motor torques (normalized to [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # Observation space: joint positions (6) + velocities (6) + torso height + velocity (2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        
        # Rendering
        self.viewer = None
        self.renderer = None
        
    def _create_mjcf_file(self):
        """Generate MJCF XML file for the robot."""
        mjcf = f"""
        <mujoco model="beewalker">
          <compiler angle="radian"/>
          <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast">
            <flag contact="enable"/>
          </option>
          
          <default>
            <joint damping="1.0" armature="0.01"/>
            <geom friction="1 0.5 0.5" condim="3"/>
          </default>
          
          <asset>
            <texture name="grid" type="2d" builtin="checker" width="512" height="512" 
                     rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
            <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
          </asset>
          
          <worldbody>
            <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
            <geom name="floor" type="plane" size="5 5 0.1" material="grid"/>
            
            <!-- Torso -->
            <body name="torso" pos="0 0 {self.config.upper_leg_length + self.config.lower_leg_length + 0.1}">
              <freejoint name="root"/>
              <geom type="box" size="{self.config.body_length/2} {self.config.body_width/2} {self.config.body_height/2}" 
                    mass="{self.config.total_mass * 0.4}" rgba="1 0.8 0 1"/>
              
              <!-- Left Leg -->
              <body name="left_hip" pos="{self.config.body_length/4} {self.config.body_width/2} 0">
                 <joint name="left_hip_joint" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                 <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -{self.config.upper_leg_length}" 
                       mass="0.1" rgba="0.2 0.2 0.8 1"/>
                 
                 <body name="left_knee" pos="0 0 -{self.config.upper_leg_length}">
                    <joint name="left_knee_joint" type="hinge" axis="0 1 0" range="-2.5 0"/>
                    <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -{self.config.lower_leg_length}" 
                          mass="0.1" rgba="0.2 0.2 0.8 1"/>
                    
                    <body name="left_foot" pos="0 0 -{self.config.lower_leg_length}">
                        <joint name="left_ankle_joint" type="hinge" axis="1 0 0" range="-0.5 0.5"/>
                        <geom type="box" size="0.05 0.03 0.01" mass="0.05" rgba="0.3 0.3 0.3 1"/>
                    </body>
                 </body>
              </body>

              <!-- Right Leg -->
              <body name="right_hip" pos="{self.config.body_length/4} -{self.config.body_width/2} 0">
                 <joint name="right_hip_joint" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                 <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -{self.config.upper_leg_length}" 
                       mass="0.1" rgba="0.8 0.2 0.2 1"/>
                 
                 <body name="right_knee" pos="0 0 -{self.config.upper_leg_length}">
                    <joint name="right_knee_joint" type="hinge" axis="0 1 0" range="-2.5 0"/>
                    <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -{self.config.lower_leg_length}" 
                          mass="0.1" rgba="0.8 0.2 0.2 1"/>
                    
                    <body name="right_foot" pos="0 0 -{self.config.lower_leg_length}">
                        <joint name="right_ankle_joint" type="hinge" axis="1 0 0" range="-0.5 0.5"/>
                        <geom type="box" size="0.05 0.03 0.01" mass="0.05" rgba="0.3 0.3 0.3 1"/>
                    </body>
                 </body>
              </body>
              
            </body>
          </worldbody>
          
          <actuator>
              <motor name="left_hip_motor" joint="left_hip_joint" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
              <motor name="left_knee_motor" joint="left_knee_joint" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
              <motor name="left_ankle_motor" joint="left_ankle_joint" gear="25" ctrllimited="true" ctrlrange="-1 1"/>
              <motor name="right_hip_motor" joint="right_hip_joint" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
              <motor name="right_knee_motor" joint="right_knee_joint" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
              <motor name="right_ankle_motor" joint="right_ankle_joint" gear="25" ctrllimited="true" ctrlrange="-1 1"/>
          </actuator>
        </mujoco>
        """
        
        # Save to file
        mjcf_path = Path(__file__).parent / "beewalker.xml"
        with open(mjcf_path, "w") as f:
            f.write(mjcf)
        return mjcf_path
    
    def _get_obs(self):
        """Get current observation."""
        # Joint positions (6)
        joint_pos = self.data.qpos[7:13].copy()  # Skip free joint (7 DOF)
        
        # Joint velocities (6)
        joint_vel = self.data.qvel[6:12].copy()   # Skip free joint (6 DOF)
        
        # Torso height and forward velocity
        torso_z = self.data.qpos[2]
        forward_vel = self.data.qvel[0]
        
        return np.concatenate([
            joint_pos,
            joint_vel,
            [torso_z, forward_vel]
        ]).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Small random perturbation
        if seed is not None:
            np.random.seed(seed)
        self.data.qpos[:] += np.random.uniform(-0.01, 0.01, self.data.qpos.shape)
        
        mujoco.mj_forward(self.model, self.data)
        
        obs = self._get_obs()
        info = {}
        
        return obs, info
    
    def step(self, action):
        # Apply actions as motor controls
        self.data.ctrl[:] = action
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get observation
        obs = self._get_obs()
        
        # Calculate reward
        torso_z = self.data.qpos[2]
        forward_vel = self.data.qvel[0]
        
        # Reward: forward progress + staying upright - energy cost
        reward = forward_vel * 1.0 + min(torso_z, 0.3) * 0.5 - 0.01 * np.sum(action ** 2)
        reward = np.clip(reward, -10.0, 10.0)
        
        # Termination: robot fell
        terminated = torso_z < 0.08
        truncated = False
        
        info = {"torso_z": torso_z, "forward_vel": forward_vel}
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


# Register the environment
gym.register(
    id="BeeWalker-v0",
    entry_point="src.gym_env:BeeWalkerEnv",
    max_episode_steps=1000,
)
