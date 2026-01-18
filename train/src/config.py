"""
BeeWalker Robot Configuration
Defines physical dimensions, mass properties, and hardware mappings.
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class RobotConfig:
    # Dimensions (meters)
    # Estimates based on standard MG996R servos and 18650 cells
    body_length: float = 0.15
    body_width: float = 0.10
    body_height: float = 0.05
    
    upper_leg_length: float = 0.10
    lower_leg_length: float = 0.10
    
    # Mass (kg)
    # MG996R ~ 55g, 18650 ~ 45g
    total_mass: float = 1.5  # Approximate total weight
    
    # Servo Limits (radians)
    servo_range: tuple = (-np.pi/2, np.pi/2)
    
    # Hardware Pin Mapping (PCA9685)
    # Left Leg (Top-Down): Hip, Knee, Ankle
    left_leg_pins: tuple = (0, 1, 2)
    
    # Right Leg (Top-Down): Hip, Knee, Ankle
    right_leg_pins: tuple = (4, 5, 6)
    
    # Control
    dt: float = 0.02  # 50Hz control loop
    
    # RL Config
    action_scale: float = 0.5
    obs_dim: int = 12  # Orientation(3) + AngularVel(3) + JointAngles(6) ? TBD
    act_dim: int = 6

ROBOT_CONFIG = RobotConfig()
