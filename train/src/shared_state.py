"""
Shared State for Real-time Visualization
Thread-safe frame buffer and stats for web dashboard.
"""

import threading
import collections


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None  # numpy array (RGB)
        self.stats = {
            "episode": 0,
            "step": 0,
            "reward": 0.0,
            "best_reward": -float('inf'),
            "status": "Initializing..."
        }
        self.history = collections.deque(maxlen=100)

    def update_frame(self, frame):
        """Update the latest frame (called from training loop)."""
        with self.lock:
            self.latest_frame = frame

    def update_stats(self, episode=None, step=None, reward=None, status=None):
        """Update training statistics."""
        with self.lock:
            if episode is not None:
                self.stats["episode"] = episode
            if step is not None:
                self.stats["step"] = step
            if reward is not None:
                self.stats["reward"] = reward
                if reward > self.stats["best_reward"]:
                    self.stats["best_reward"] = reward
                self.history.append(reward)
            if status is not None:
                self.stats["status"] = status

    def get_frame(self):
        """Get the latest frame (called from web server)."""
        with self.lock:
            return self.latest_frame

    def get_stats(self):
        """Get current stats and history."""
        with self.lock:
            return self.stats.copy(), list(self.history)


# Global singleton
shared_state = SharedState()
