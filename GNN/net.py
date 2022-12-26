import torch
import torch.nn as nn

from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
# from dgllife.model import AttentiveFPPredictor

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl




# GSGAT
class FPSAT(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 self_feats_dim,
                 linear_feats_len,
                 n_tasks,
                 dropout,
                 num_layers,
                 num_timesteps,
                 graph_feat_size
                 ):
        super(FPSAT, self).__init__()

        self.gnn = AttentiveFPGNN(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            graph_feat_size=graph_feat_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.readout = AttentiveFPReadout(
            feat_size=graph_feat_size,
            dropout=dropout,
            num_timesteps=num_timesteps

        )

        self.fp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self_feats_dim, linear_feats_len),
        )

        self.predict = nn.Sequential(
            # nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(graph_feat_size + linear_feats_len, n_tasks))

    def forward(self, graph, node_feats, edge_feats, self_feats, get_node_weight=None):
        node_feats = self.gnn(graph, node_feats, edge_feats)
        graph_feats = self.readout(graph, node_feats, get_node_weight)

        new_feats = torch.cat((graph_feats, self_feats), dim=1)

        pred = self.predict(new_feats)

        return pred

# Attentive FP

class AttentiveFPPredictor(nn.Module):
    """AttentiveFP for regression and classification on graphs.
    AttentiveFP is introduced in
    `Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism. <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    num_timesteps : int
        Times of updating the graph representations with GRU. Default to 2.
    graph_feat_size : int
        Size for the learned graph representations. Default to 200.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    dropout : float
        Probability for performing the dropout. Default to 0.
    """
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 num_timesteps=2,
                 graph_feat_size=200,
                 n_tasks=1,
                 dropout=0.):
        super(AttentiveFPPredictor, self).__init__()

        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats, get_node_weight=False):
        """Graph-level regression/soft classification.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.
        get_node_weight : bool
            Whether to get the weights of atoms during readout. Default to False.
        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        """
        node_feats = self.gnn(g, node_feats, edge_feats)
        if get_node_weight:
            g_feats, node_weights = self.readout(g, node_feats, get_node_weight)
            return self.predict(g_feats), node_weights
        else:
            g_feats = self.readout(g, node_feats, get_node_weight)
            return self.predict(g_feats)


# EGAT

class EGAT(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 self_feats_dim,
                 linear_feats_len,
                 n_tasks,
                 dropout,
                 num_layers,
                 num_timesteps,
                 graph_feat_size
                 ):
        super(EGAT, self).__init__()

        self.gnn = AttentiveFPGNN(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            graph_feat_size=graph_feat_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.readout = AttentiveFPReadout(
            feat_size=graph_feat_size,
            dropout=dropout,
            num_timesteps=num_timesteps

        )

        self.fp = nn.Sequential(
            nn.Dropout(dropout),

            nn.Linear(self_feats_dim, linear_feats_len),
        )

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            #
            # nn.ReLU(),

            nn.Linear(graph_feat_size + linear_feats_len, n_tasks))

    def forward(self, graph, node_feats, edge_feats, self_feats, get_node_weight=None):
        node_feats = self.gnn(graph, node_feats, edge_feats)
        graph_feats = self.readout(graph, node_feats, get_node_weight)

        new_feats = torch.cat((graph_feats, self_feats), dim=1)

        pred = self.predict(new_feats)

        return pred



# FGAT_SaN

class FGAT_SaN(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 self_feats_dim,
                 linear_feats_len,
                 n_tasks,
                 dropout,
                 num_layers,
                 num_timesteps,
                 graph_feat_size
                 ):
        super(FGAT_SaN, self).__init__()

        self.gnn = AttentiveFPGNN(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            graph_feat_size=graph_feat_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.readout = AttentiveFPReadout(
            feat_size=graph_feat_size,
            dropout=dropout,
            num_timesteps=num_timesteps

        )

        self.fp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self_feats_dim, linear_feats_len),
        )

        self.predict = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * (graph_feat_size + linear_feats_len), n_tasks))

    def forward(self, graph1, graph2, node_feats, edge_feats, node_feats2, edge_feats2, self_feats1, self_feats2,
                get_node_weight=None):
        node_feats1 = self.gnn(graph1, node_feats, edge_feats)
        graph_feats1 = self.readout(graph1, node_feats1, get_node_weight)
        new_feats1 = torch.cat((graph_feats1, self_feats1), dim=1)

        node_feats2 = self.gnn(graph2, node_feats2, edge_feats2)
        graph_feats2 = self.readout(graph2, node_feats2, get_node_weight)
        new_feats2 = torch.cat((graph_feats2, self_feats2), dim=1)

        new_feats = torch.cat((new_feats1, new_feats2), dim=1)

        pred = self.predict(new_feats)

        return pred

# EGCN

msg = fn.copy_src(src='h', out='m')

def reduce(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)

    return {'h': accum}


class NodeApplyModule(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, node):
        h = self.linear(node.data['h'])

        return {'h': h}


class GCNLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GCNLayer, self).__init__()
        self.apply_mod = NodeApplyModule(dim_in, dim_out)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)

        return g.ndata.pop('h')


class Net(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super(Net, self).__init__()

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)
        self.fc1 = nn.Linear(20 + dim_self_feat, 10)
        self.fc2 = nn.Linear(10, dim_out)

    def forward(self, g, self_feat):
        h = F.relu(self.gc1(g, g.ndata['feat'])) #传入图 和 特征矩阵
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')
        hg = torch.cat((hg, self_feat), dim=1)  #作者的改进之处

        out = F.relu(self.fc1(hg))
        out = self.fc2(out)

        return out
