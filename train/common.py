"""Shared state between training thread and web server."""

import threading
import collections


class SharedState:
    """Thread-safe container for sharing frames and stats between threads."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None  # numpy array (RGB)
        self.stats = {
            "episode": 0,
            "reward": 0.0,
            "steps": 0,
            "best_reward": -float('inf'),
            "training": True
        }
        self.history = collections.deque(maxlen=100)  # Last 100 episode rewards

    def update_frame(self, frame):
        """Update the latest frame for streaming."""
        with self.lock:
            self.latest_frame = frame

    def update_stats(self, episode, reward, steps):
        """Update training statistics."""
        with self.lock:
            self.stats["episode"] = episode
            self.stats["reward"] = reward
            self.stats["steps"] = steps
            if reward > self.stats["best_reward"]:
                self.stats["best_reward"] = reward
            self.history.append(reward)

    def get_frame(self):
        """Get the latest frame for streaming."""
        with self.lock:
            return self.latest_frame

    def get_stats(self):
        """Get current stats and history."""
        with self.lock:
            return self.stats.copy(), list(self.history)


# Global instance shared between modules
shared_state = SharedState()
