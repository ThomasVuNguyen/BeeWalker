"""PPO Agent for BeeWalker training."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


class PPOAgent:
    """
    Proximal Policy Optimization agent.
    
    Features:
        - Clipped surrogate objective
        - Value function loss
        - Entropy bonus for exploration
        - Checkpointing support
    """
    
    def __init__(self, policy, lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 eps_clip=0.2, K_epochs=4, entropy_coef=0.01, value_coef=0.5):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        self.mse_loss = nn.MSELoss()
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
    
    def select_action(self, state):
        """
        Select action given current state.
        
        Args:
            state: numpy array of shape (obs_dim,)
            
        Returns:
            action: numpy array of shape (act_dim,)
            logprob: log probability of the action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_mean, action_std, value = self.policy(state_tensor)
            
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(dim=-1)
            
        return (
            action.detach().cpu().numpy()[0],
            action_logprob.detach().cpu().numpy()[0],
            value.detach().cpu().numpy()[0, 0]
        )
    
    def store_transition(self, state, action, logprob, reward, done, value):
        """Store a transition in the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(done)
        self.values.append(value)
    
    def clear_buffer(self):
        """Clear the experience buffer."""
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
    
    def _compute_gae(self):
        """Compute Generalized Advantage Estimation."""
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.is_terminals)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self):
        """Update policy using PPO."""
        if len(self.states) == 0:
            return 0.0
        
        # Compute advantages and returns
        advantages, returns = self._compute_gae()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        old_states = torch.FloatTensor(np.array(self.states))
        old_actions = torch.FloatTensor(np.array(self.actions))
        old_logprobs = torch.FloatTensor(np.array(self.logprobs))
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        total_loss = 0.0
        
        # PPO update for K epochs
        for _ in range(self.K_epochs):
            # Get current policy outputs
            action_mean, action_std, state_values = self.policy(old_states)
            state_values = state_values.squeeze()
            
            # Calculate log probabilities
            dist = torch.distributions.Normal(action_mean, action_std)
            logprobs = dist.log_prob(old_actions).sum(dim=1)
            entropy = dist.entropy().sum(dim=1)
            
            # Ratio for PPO
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Clipped surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = self.mse_loss(state_values, returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.clear_buffer()
        return total_loss / self.K_epochs
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, weights_only=True)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded checkpoint from {filepath}")
            return True
        return False
