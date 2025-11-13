import torch
import torch.nn as nn
import torch.nn.functional as F

class MeasurementEmbedder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [B, input_dim] (numeric values like [height, width, radius])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Return as [B, 1, hidden_dim] → behaves like a "token"
        return x.unsqueeze(1)
