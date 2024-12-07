import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical


class Agent(nn.Module):
    def __init__(self, num_actions, comm_size, device, num_pistons, comm_size_compressed=32, neighbor_range=None):
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Conv2d(4, 32, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
        )
        self.comm_size_compressed = comm_size_compressed
        self.num_pistons = num_pistons
        self.device = device
        self.comm_aggregator = MultiAgentAttention(input_dim=comm_size, hidden_dim=8, output_dim=self.comm_size_compressed, n_heads=4, neighbor_range=neighbor_range)

        # Want to give actor/critic more repr power (maybe encoder/decoder)
        self.actor = nn.Sequential(
            self._layer_init(nn.Linear(512 + self.comm_size_compressed + 1, 128), std=0.01),
            nn.ReLU(),
            self._layer_init(nn.Linear(128, 64), std=0.01),
            nn.ReLU(),
            self._layer_init(nn.Linear(64, num_actions), std=0.01))

        self.critic = nn.Sequential(
            self._layer_init(nn.Linear(512 + self.comm_size_compressed + 1, 128)),
            nn.ReLU(),
            self._layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            self._layer_init(nn.Linear(64, 1), std=0.01))

        self.prev_action = None

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x, comm=None):
        if comm is None:
            comm = torch.zeros((self.num_pistons, self.comm_size_compressed)).to(device)
        hidden = self.network(x / 255.0)
        critic_input = torch.concatenate([hidden, comm], dim=1)
        return self.critic(critic_input)

    def get_action_and_value(self, x, id, action=None, comm=None):
        B, _, _, _ = x.shape
        if comm is None:
            comm = torch.zeros((B, self.comm_size_compressed)).to(self.device)
        hidden = self.network(x / 255.0)
        actor_input = torch.concatenate([hidden, comm, id], dim=1)
        logits = self.actor(actor_input)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        self.prev_action = action
        return action, probs.log_prob(action), probs.entropy(), self.critic(actor_input)

    def get_comm(self, x, id):
        with torch.no_grad():
            hidden = self.network(x / 255.0)
        comm_input = torch.concatenate([hidden, id], dim=1)
        comm = self.comm_aggregator(comm_input)
        return comm

    def reset_comm(self):
        self.prev_action = None
        
class MultiAgentAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, neighbor_range=None):
        super(MultiAgentAttention, self).__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.neighbor_range = neighbor_range  # Max number of right-side neighbors to attend
        # Attention layers
        self.query = nn.Linear(input_dim, hidden_dim * n_heads)
        self.key = nn.Linear(input_dim, hidden_dim * n_heads)
        self.value = nn.Linear(input_dim, hidden_dim * n_heads)
        
        assert hidden_dim * n_heads == output_dim, "Output dim must be equal to hidden_dim * n_heads"

    def forward(self, agent_features, return_attention=False):
        """
        agent_features: Tensor of shape [batch_size (n_agents), input_dim]
        """
        batch_size, input_dim = agent_features.shape

        # Compute query, key, value
        query = self.query(agent_features).view(batch_size, self.n_heads, -1)
        key = self.key(agent_features).view(batch_size, self.n_heads, -1)
        value = self.value(agent_features).view(batch_size, self.n_heads, -1)

        # Compute attention scores
        scores = torch.einsum("bhd,chd->bhc", query, key) / (self.hidden_dim ** 0.5)

        # Apply Causal Mask
        if self.neighbor_range is not None:
            mask = self._generate_causal_mask(batch_size, self.neighbor_range, agent_features.device)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        attended_features = torch.einsum("bhc,chd->bhd", attention_weights, value)
        attended_features = attended_features.reshape(batch_size, -1)

        if return_attention:
            return attended_features, attention_weights
        return attended_features

    def _generate_causal_mask(self, batch_size, neighbor_range, device):
        """
        Generate a mask where only right-side neighbors up to neighbor_range are allowed.
        Shape: [batch_size, batch_size]
        """
        mask = torch.zeros((batch_size, batch_size), device=device)
        for i in range(batch_size):
            start = i
            end = min(batch_size, i + 1 + neighbor_range)
            mask[i, start:end] = 1
        return mask.unsqueeze(1).expand(-1, self.n_heads, -1)