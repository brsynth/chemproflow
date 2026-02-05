import torch
from torch import nn
from torch_geometric.nn import global_add_pool, MessagePassing

class DMPNNLayer(MessagePassing):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim):
        super().__init__(
            aggr="mean", flow="source_to_target"
        )  # Directed message passing
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_in_dim + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_update = nn.Linear(node_in_dim + hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: source node features, edge_attr: bond features
        return self.edge_mlp(torch.cat([x_j, edge_attr], dim=1))

    def update(self, aggr_out, x):
        return self.node_update(torch.cat([x, aggr_out], dim=1))
    
class ChemEmbedding(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super().__init__()
        self.gnn1 = DMPNNLayer(node_feat_dim, edge_feat_dim, hidden_dim)
        self.gnn2 = DMPNNLayer(hidden_dim, edge_feat_dim, hidden_dim)
        self.gnn3 = DMPNNLayer(hidden_dim, edge_feat_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        x = self.gnn1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.gnn2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.gnn3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = torch.relu(x)
        # Pool node features to get graph representations
        return global_add_pool(x, batch)
    
class ChemReadout(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, graph_emb):
        return self.readout(graph_emb)
