"""
BeeWalker Simulation Environment using Brax
"""

import brax
from brax import envs
from brax.io import html
import jax.numpy as jnp
from src.config import ROBOT_CONFIG

class BeeWalkerEnv(envs.Env):
    def __init__(self, config=None):
        self.robot_config = config or ROBOT_CONFIG
        
        # Define the system config (MJCF-like structure)
        # This is a placeholder for the actual kinematic tree construction
        # In a real implementation we would generate a MJCF string here
        mjcf_str = self._create_system()
        self.sys = brax.io.mjcf.loads(mjcf_str)
        
    def _create_system(self):
        # Todo: Generate MJCF string string based on config
        # For now, we will use a standard ant/humanoid as a placeholder 
        # to ensure the pipeline works, then customize it.
        # But since we need specific joints, we should construct the MJCF.
        
        # Simplified MJCF construction
        mjcf = f"""
        <mujoco>
          <compiler angle="radian"/>
          <option timestep="{self.robot_config.dt}"/>
          <asset>
              <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
              <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
          </asset>
          <worldbody>
            <light pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>
            <geom name="floor" type="plane" size="0 0 0.1" material="grid"/>
            
            <!-- Torso -->
            <body name="torso" pos="0 0 {self.robot_config.upper_leg_length + self.robot_config.lower_leg_length + 0.1}">
              <geom type="box" size="{self.robot_config.body_length/2} {self.robot_config.body_width/2} {self.robot_config.body_height/2}" mass="{self.robot_config.total_mass * 0.4}"/>
              <joint name="root" type="free"/>
              
              <!-- Left Leg -->
              <body name="left_hip" pos="{self.robot_config.body_length/2} {self.robot_config.body_width/2} 0">
                 <joint name="left_hip_joint" axis="0 1 0" range="-1.57 1.57"/>
                 <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -{self.robot_config.upper_leg_length}"/>
                 
                 <body name="left_knee" pos="0 0 -{self.robot_config.upper_leg_length}">
                    <joint name="left_knee_joint" axis="0 1 0" range="-1.57 1.57"/>
                    <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -{self.robot_config.lower_leg_length}"/>
                    
                    <body name="left_ankle" pos="0 0 -{self.robot_config.lower_leg_length}">
                        <joint name="left_ankle_joint" axis="1 0 0" range="-0.5 0.5"/> 
                        <geom type="box" size="0.05 0.03 0.01"/>
                    </body>
                 </body>
              </body>

              <!-- Right Leg -->
              <body name="right_hip" pos="{self.robot_config.body_length/2} -{self.robot_config.body_width/2} 0">
                 <joint name="right_hip_joint" axis="0 1 0" range="-1.57 1.57"/>
                 <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -{self.robot_config.upper_leg_length}"/>
                 
                 <body name="right_knee" pos="0 0 -{self.robot_config.upper_leg_length}">
                    <joint name="right_knee_joint" axis="0 1 0" range="-1.57 1.57"/>
                    <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -{self.robot_config.lower_leg_length}"/>
                    
                    <body name="right_ankle" pos="0 0 -{self.robot_config.lower_leg_length}">
                        <joint name="right_ankle_joint" axis="1 0 0" range="-0.5 0.5"/> 
                        <geom type="box" size="0.05 0.03 0.01"/>
                    </body>
                 </body>
              </body>
              
            </body>
          </worldbody>
          <actuator>
              <motor name="left_hip_motor" joint="left_hip_joint" gear="100" ctrllimited="true" ctrlrange="-1 1"/>
              <motor name="left_knee_motor" joint="left_knee_joint" gear="100" ctrllimited="true" ctrlrange="-1 1"/>
              <motor name="left_ankle_motor" joint="left_ankle_joint" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
              <motor name="right_hip_motor" joint="right_hip_joint" gear="100" ctrllimited="true" ctrlrange="-1 1"/>
              <motor name="right_knee_motor" joint="right_knee_joint" gear="100" ctrllimited="true" ctrlrange="-1 1"/>
              <motor name="right_ankle_motor" joint="right_ankle_joint" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
          </actuator>
        </mujoco>
        """
        
        # Note: We need to use brax.io.mjcf.load_model usually
        # But Brax's API changes often. We will assume standard load.
        # This is strictly a placeholder implementation.
        return mjcf

        
    def reset(self, rng):
        q = self.sys.init_q
        qd = jnp.zeros(self.sys.qd_size())
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done = jnp.array(0.0), jnp.array(0.0)
        metrics = {}
        return brax.envs.State(pipeline_state, obs, reward, done, metrics)

    def step(self, state, action):
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)
        
        # Get torso position and velocity (first body after free joint)
        # pipeline_state.x.pos is (num_bodies, 3), x.rot is (num_bodies, 4)
        torso_pos = pipeline_state.x.pos[0]  # (3,)
        torso_z = torso_pos[2]
        
        # Forward velocity: first 6 dofs are the free joint (x,y,z,rx,ry,rz)
        forward_vel = pipeline_state.qd[0]  # x velocity
        
        # Simple reward: forward motion + staying upright - energy
        reward = forward_vel * 0.5 + torso_z * 0.5 - 0.01 * jnp.sum(action ** 2)
        
        # Guard against NaN and clamp to reasonable range
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        reward = jnp.clip(reward, -10.0, 10.0)
        
        # Done if fallen
        done = jnp.where(torso_z < 0.1, 1.0, 0.0)
        
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state):
        # Observation: joint angles, velocities, body orientation
        return jnp.concatenate([
            pipeline_state.q, 
            pipeline_state.qd,
            pipeline_state.x.pos[0],
            pipeline_state.x.rot[0]
        ])
        
    def pipeline_init(self, q, qd):
        from brax.mjx import pipeline
        return pipeline.init(self.sys, q, qd)

    @property
    def observation_size(self):
        # q + qd + pos + rot
        return self.sys.q_size() + self.sys.qd_size() + 3 + 4

    @property
    def action_size(self):
        return self.sys.act_size()

    @property
    def backend(self):
        return 'mjx'

    def pipeline_step(self, pipeline_state, action):
        from brax.mjx import pipeline
        return pipeline.step(self.sys, pipeline_state, action)
