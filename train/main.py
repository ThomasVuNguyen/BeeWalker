"""BeeWalker Training - Main Entry Point.

Starts the Flask web server and training loop.
Web dashboard available at http://localhost:1306
"""

import os
import sys
import threading

# Set MuJoCo rendering backend for headless mode
os.environ.setdefault('MUJOCO_GL', 'egl')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.server import run_server
from training.train import train


def main():
    """Start web server and training."""
    print("=" * 60)
    print("  BeeWalker Training System")
    print("  Web Dashboard: http://localhost:1306")
    print("=" * 60)
    print()
    
    # Start web server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Give server a moment to start
    import time
    time.sleep(1)
    
    # Run training in main thread
    try:
        train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining error: {e}")
        raise


if __name__ == "__main__":
    main()
