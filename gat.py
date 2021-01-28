import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch import nn
from tqdm import tqdm
from time import time

class AttentionGraphModel(nn.Module):

    def __init__(self, g, n_layers, n_head, input_size, hidden_size, output_size, nonlinearity, device):
        """
        Highly inspired from https://arxiv.org/pdf/1710.10903.pdf
        """
        super().__init__()

        self.n_head = n_head
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.device = device
        self.g = g
        self.activation = nonlinearity
        self.linears = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        self.layers = [] # So that train still works, basically what we say is that we don't use dgl GraphConv

        self.add_attention_layer(input_size, hidden_size, n_head)

        for i in range(n_layers - 1):
            self.add_attention_layer(hidden_size*n_head, hidden_size, n_head)

        self.add_attention_layer(hidden_size*n_head, output_size, n_head)
    
    def add_attention_layer(self, n_features, n_hidden_features, n_head):
        linear = Linear(n_features, n_hidden_features)
        attn_layer = Conv1d(n_hidden_features*2*n_head, n_head, kernel_size=1, groups=n_head)
        
        self.linears.append(linear)
        self.attn_layers.append(attn_layer)

    def forward(self, inputs):
        outputs = inputs
        for i, (linear, attn_layer) in enumerate(zip(self.linears, self.attn_layers)):
            outputs = self.forward_attention(outputs, linear, attn_layer, final_layer=(i==self.n_layers))
        return outputs

    def get_h2(self, attn_layer, h, indices, adjacency_list):
        """
        h2 is of size (n_nodes,n_hidden_features,n_head)
        """
        n_nodes, n_features = h.shape
        n_edges = len(indices[0])

        # values = torch.zeros(n_edges, self.n_head, device=self.device)
        h_conv = h.repeat(1,self.n_head).view(n_nodes,1,n_features*self.n_head,1)
        h_concat = torch.cat([torch.cat((h_conv[i],h_conv[j]), dim=1) for (i,j) in zip(indices[0], indices[1])], dim=0)
        values = attn_layer(h_concat).view(n_edges, -1)
        
        # for k, (i,j) in enumerate(zip(indices[0], indices[1])):
        #     values[k,:] = attn_layer(torch.cat((h_conv[i],h_conv[j]), dim=1)).view(-1)
        e = torch.sparse_coo_tensor(indices,values)

        alpha = torch.sparse.softmax(e, dim=1).coalesce()
        values = alpha.values()
    
        t = time()
        h2 = torch.zeros(n_nodes, n_features, self.n_head, device=self.device)
        for i,j,v in zip(indices[0], indices[1], values):
            h2[i,:,:] += h[j,:].unsqueeze(-1) @ v.unsqueeze(0)
        print(time()-t)
        t = time()
        h2 = torch.zeros(n_nodes, n_features, self.n_head, device=self.device)
        h = h.unsqueeze(2) # (n_nodes, n_features, 1)
        values = values.unsqueeze(1) # (n_edges, 1, n_head)
        for i in range(n_nodes):
            h_neigh = torch.cat([h[j,:] for j in adjacency_list[i]], dim=1)
            values_neigh = torch.cat([values[j,:] for j in adjacency_list[i]], dim=0)
            h2[i] = h_neigh @ values_neigh
        print(time()-t)

        return h2

    def forward_attention(self, x, linear, attn_layer, final_layer=False):
        h = F.leaky_relu(linear(x), negative_slope=0.2)

        indices, adjacency_list = self.get_graph_structure()

        h2 = self.get_h2(attn_layer, h, indices, adjacency_list)
        
        if final_layer:
            return self.activation(h2.mean(dim=-1))
        else:
            return self.activation(h2.view(h2.size(0),-1))


    def get_graph_structure(self):
        adjacency_matrix = self.g.adjacency_matrix().coalesce().to(self.device)
        indices = adjacency_matrix.indices()
        n_nodes = self.g.num_nodes()

        adjacency_list = [[]] * n_nodes
        for (i,j) in zip(indices[0], indices[1]):
            adjacency_list[i].append(j)

        return indices, adjacency_list