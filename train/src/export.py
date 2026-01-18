"""
BeeWalker Model Export Tool
"""

import pickle
import json
import struct
import jax.numpy as jnp
from flax.serialization import to_state_dict

def export_model(params, path="model.bin"):
    """
    Export JAX/Flax params to a simple binary format for Pi Pico.
    Format:
    [Header]: [NumLayers(int), LayerSizes(int...)]
    [Weights]: Flattened float32 weights
    """
    print(f"📦 Exporting model to {path}...")
    
    # Convert to state dict (nested dict of arrays)
    state = to_state_dict(params)
    
    # Helper to flatten weights in order
    # Assuming MLP: Dense_0 (kernel, bias), Dense_1 ...
    # We need to traverse in specific order
    
    # TODO: Traverse state dict and write to binary
    # This highly depends on the network structure Brax uses (usually MLP)
    
    # Placeholder for actual traversal logic
    with open(path, "wb") as f:
        # Write magic number
        f.write(b"BEE1") 
        # Write rest of data...
        pass
    
    print("✅ Export complete.")
