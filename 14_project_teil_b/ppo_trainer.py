"""
PPO Training Module for Trading

Implements:
- Actor-Critic architecture
- GAE (Generalized Advantage Estimation)
- PPO with clipping
- Training loop with evaluation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym


class ActorCritic(nn.Module):
    """Actor-Critic network for continuous control."""

    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()

        # Shared feature extractor
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        # Actor head (policy)
        self.mu = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -1.0)

        # Critic head (value function)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, obs):
        """
        Returns distribution and value estimate.

        Parameters
        ----------
        obs : torch.Tensor
            Observations [batch_size, obs_dim]

        Returns
        -------
        dist : Normal
            Policy distribution
        value : torch.Tensor
            Value estimates [batch_size]
    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu(x)
        std = torch.exp(self.log_std)
        
        # Clip std for numerical stability
        std = torch.clamp(std, min=1e-5, max=10.0)
        
        # Ensure mu is finite
        mu = torch.clamp(mu, min=-100, max=100)
        
        dist = Normal(mu, std)
        value = self.v(x).squeeze(-1)
        return dist, value

    def get_action_and_value(self, obs):
        """Get action and value for rollout."""
        dist, value = self.forward(obs)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value

    def get_value(self, obs):
        """Get value only (for bootstrap)."""
        with torch.no_grad():
            _, value = self.forward(obs)
        return value


def squash_action(u, low=-1.0, high=1.0):
    """
    Squash unbounded action using tanh.

    Parameters
    ----------
    u : torch.Tensor
        Unbounded action from distribution
    low : float
    high : float
        Bounds for output

    Returns
    -------
    torch.Tensor
        Bounded action
    """
    return torch.tanh(u) * (high - low) / 2.0 + (high + low) / 2.0


def logprob_squashed(dist, u, low=-1.0, high=1.0):
    """
    Compute log probability of squashed action.

    Accounts for change of variables:
    log p(a) = log p(u) - log|det(Jacobian)|
    where a = tanh(u) and det(Jacobian) = prod(1 - tanh(u)^2)
    """
    logp_u = dist.log_prob(u).sum(-1)
    a = torch.tanh(u)
    eps = 1e-6
    log_det = torch.log(1.0 - a.pow(2) + eps).sum(-1)
    return logp_u - log_det


def compute_gae(rewards, dones, values, last_value, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation.

    Parameters
    ----------
    rewards : torch.Tensor
        [T, N] trajectory rewards
    dones : torch.Tensor
        [T, N] done flags
    values : torch.Tensor
        [T, N] value estimates
    last_value : torch.Tensor
        [N] bootstrap value
    gamma : float
        Discount factor
    lam : float
        GAE lambda

    Returns
    -------
    returns, advantages
    """
    T, N = rewards.shape
    adv = torch.zeros(T, N, device=values.device)
    gae = torch.zeros(N, device=values.device)

    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        adv[t] = gae

    returns = adv + values
    return returns, adv


class PPOTrainer:
    """PPO trainer for continuous control."""

    def __init__(
        self,
        obs_dim,
        act_dim,
        device,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.001,
        max_grad_norm=0.5,
        target_kl=0.1,
    ):
        """
        Parameters
        ----------
        obs_dim : int
            Observation dimension
        act_dim : int
            Action dimension
        device : torch.device
        lr : float
            Learning rate
        gamma : float
            Discount factor
        gae_lambda : float
            GAE lambda
        clip_eps : float
            PPO clip parameter
        vf_coef : float
            Value function loss coefficient
        ent_coef : float
            Entropy bonus coefficient
        max_grad_norm : float
            Gradient clipping
        target_kl : float
            Early stopping target KL
        """
        self.device = device
        self.model = ActorCritic(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

    def rollout(self, env, num_steps, action_bounds=None):
        """
        Collect rollout from environment.

        Parameters
        ----------
        env : gym.vector.VectorEnv
        num_steps : int
        action_bounds : tuple, optional
            (low, high) for action squashing

        Returns
        -------
        dict with rollout buffers
        """
        if action_bounds is None:
            action_bounds = (-1.0, 1.0)

        obs, _ = env.reset() if not hasattr(env, '_obs') else (env._obs, {})
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        num_envs = env.num_envs
        obs_dim = env.single_observation_space.shape[0]
        act_dim = env.single_action_space.shape[0]

        # Initialize buffers
        obs_buf = torch.zeros(num_steps, num_envs, obs_dim, device=self.device)
        u_buf = torch.zeros(num_steps, num_envs, act_dim, device=self.device)
        a_buf = torch.zeros(num_steps, num_envs, act_dim, device=self.device)
        logp_buf = torch.zeros(num_steps, num_envs, device=self.device)
        rew_buf = torch.zeros(num_steps, num_envs, device=self.device)
        done_buf = torch.zeros(num_steps, num_envs, device=self.device)
        val_buf = torch.zeros(num_steps, num_envs, device=self.device)

        ep_returns = np.zeros(num_envs, dtype=np.float32)
        ep_history = []

        # Rollout
        for t in range(num_steps):
            obs_buf[t] = obs

            with torch.no_grad():
                dist, value = self.model(obs)
                u = dist.sample()
                a = squash_action(u, action_bounds[0], action_bounds[1])
                logp = logprob_squashed(dist, u, action_bounds[0], action_bounds[1])

            u_buf[t] = u
            a_buf[t] = a
            logp_buf[t] = logp.detach()
            val_buf[t] = value.detach()

            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(a.detach().cpu().numpy())
            done = np.logical_or(terminated, truncated)

            rew_buf[t] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            done_buf[t] = torch.as_tensor(terminated, dtype=torch.float32, device=self.device)

            # Track episode returns
            ep_returns += reward
            if done.any():
                finished = np.where(done)[0]
                ep_history.extend(ep_returns[finished].tolist())
                ep_returns[finished] = 0.0

            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)

        # Bootstrap last value
        with torch.no_grad():
            _, last_value = self.model(obs)

        # Compute returns and advantages
        returns, adv = compute_gae(
            rew_buf, done_buf, val_buf, last_value,
            gamma=self.gamma, lam=self.gae_lambda
        )

        return {
            "obs": obs_buf,
            "u": u_buf,
            "a": a_buf,
            "logp": logp_buf,
            "returns": returns,
            "advantages": adv,
            "values": val_buf,
            "ep_history": ep_history,
        }, obs

    def update(self, rollout, num_epochs, minibatch_size, action_bounds=None):
        """
        PPO update step.

        Parameters
        ----------
        rollout : dict
            Rollout buffers from self.rollout()
        num_epochs : int
            Number of training epochs
        minibatch_size : int
        action_bounds : tuple, optional

        Returns
        -------
        dict with training metrics
        """
        if action_bounds is None:
            action_bounds = (-1.0, 1.0)

        # Flatten buffers
        obs = rollout["obs"]
        u = rollout["u"]
        old_logp = rollout["logp"]
        old_value = rollout["values"]
        returns = rollout["returns"]
        advantages = rollout["advantages"]

        B = obs.shape[0] * obs.shape[1]
        obs = obs.reshape(B, -1)
        u = u.reshape(B, -1)
        old_logp = old_logp.reshape(B)
        old_value = old_value.reshape(B)
        returns = returns.reshape(B).detach()
        advantages = advantages.reshape(B).detach()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create indices for minibatches
        idx = torch.arange(B, device=self.device)

        metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "approx_kl": [],
        }

        for epoch in range(num_epochs):
            perm = idx[torch.randperm(B, device=self.device)]

            for start in range(0, B, minibatch_size):
                mb_idx = perm[start : start + minibatch_size]

                # Get model outputs
                dist, value = self.model(obs[mb_idx])
                logp = logprob_squashed(dist, u[mb_idx], action_bounds[0], action_bounds[1])
                entropy = dist.entropy().sum(-1)

                # Compute losses
                ratio = torch.exp(logp - old_logp[mb_idx])

                unclipped = ratio * advantages[mb_idx]
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[mb_idx]
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = (returns[mb_idx] - value).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    approx_kl = (old_logp[mb_idx] - logp).mean().item()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy_loss"].append(entropy_loss.item())
                metrics["approx_kl"].append(approx_kl)

                # Early stopping
                if approx_kl > self.target_kl:
                    break

            # Clamp std to reasonable range
            with torch.no_grad():
                self.model.log_std.clamp_(-2.0, -0.5)

        # Average metrics
        metrics = {k: np.mean(v) for k, v in metrics.items()}

        return metrics

    def save(self, path):
        """Save model."""
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Load model."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))


if __name__ == "__main__":
    print("PPO Module imported successfully!")

