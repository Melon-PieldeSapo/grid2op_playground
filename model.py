import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, conv1_units=64, fc1_units=64, fc2_units=64,
                 fc3_units=64, fc4_units=64, fc5_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = GCNConv(state_size, conv1_units)
        self.conv2 = GCNConv(conv1_units, fc1_units)
        self.fc1 = nn.Linear(fc1_units, fc2_units)
        self.fc2 = nn.Linear(fc2_units, fc3_units)
        self.fc3 = nn.Linear(fc3_units, fc4_units)
        self.fc4 = nn.Linear(fc4_units, fc5_units)
        self.fc5 = nn.Linear(fc5_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x, edge_index = state.x, state.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))

        return F.relu(self.fc5(x))
