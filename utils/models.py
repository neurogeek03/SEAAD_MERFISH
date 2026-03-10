import torch_geometric.nn as gnn
from torch import nn

class SimpleGCN(nn.Module):
    
    def __init__(
        self, 
        num_features: int, 
        hidden_dim: int, 
        num_classes: int,
        dropout_prob=0.2
    ):
        super().__init__()
        self.conv_in = gnn.GCNConv(num_features, hidden_dim)
        self.conv_mid = gnn.GCNConv(hidden_dim, hidden_dim)
        ratio = (num_classes / hidden_dim) ** (1 / 3)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * ratio)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(int(hidden_dim * ratio), int(hidden_dim * ratio ** 2)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(int(hidden_dim * ratio ** 2), num_classes),
            nn.ReLU()
        )
    
    def forward(self, x, edge_index, batch):
        x = self.conv_in(x, edge_index)
        x = x.relu()
        x = self.conv_mid(x, edge_index)
        x = x.relu()
        x = self.conv_mid(x, edge_index)
        x = x.relu()
        x = self.conv_mid(x, edge_index)
        x = x.relu()
        x = gnn.global_mean_pool(x, batch)
        x = self.mlp(x)
        
        return x
        


class Simple(nn.Module):
    
    def __init__(self, x_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 2),
            nn.GELU(),
            nn.Linear(2, 5),
            nn.GELU()
        )
    
    def forward(self, x):
        
        return self.model(x)


def get_model(name: str, **kwargs):
    
    configured_models = [
        'simple',
        'simple-gcn'
    ]
    
    if name not in configured_models:
        raise NotImplementedError(f'{name} is not a configured model, `name` must be one of {configured_models}')
    
    if name == 'simple':
        
        return Simple(**kwargs)

    elif name == 'simple-gcn':
        
        return SimpleGCN(**kwargs)
    