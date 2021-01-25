import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch import nn
from tqdm import tqdm

class AttentionGraphModel(nn.Module):

    def __init__(self, g, n_layers, input_size, hidden_size, output_size, nonlinearity, device):
        """
        Highly inspired from https://arxiv.org/pdf/1710.10903.pdf
        """
        super().__init__()

        self.device = device
        self.g = g
        self.activation = nonlinearity
        self.linears = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        self.layers = [] # So that train still works, basically what we say is that we don't use dgl GraphConv

        self.add_simple_attention_layer(input_size, hidden_size)

        for i in range(n_layers - 1):
            self.add_simple_attention_layer(hidden_size, hidden_size)

        self.add_simple_attention_layer(hidden_size, output_size)

    def forward(self, inputs):
        outputs = inputs
        for linear, attn in zip(self.linears, self.attn_layers):
            outputs = self.forward_attention(outputs, linear, attn)
        return outputs

    def forward_attention_dense(self, x, linear, attn, final_layer=False):
        n_nodes = x.shape[0]
        h = F.leaky_relu(linear(x), negative_slope=0.2)    
        adjacency_matrix = self.g.adjacency_matrix().coalesce()

        big_h_left = torch.stack([torch.stack([h[i]]*n_nodes, dim=0) for i in range(n_nodes)])
        big_h_right = torch.stack([h]*n_nodes, dim=0)
        big_h = torch.stack([big_h_left, big_h_right], dim=1)
        e = attn(big_h).view(n_nodes, n_nodes)

        e_neighb = torch.sum(torch.exp(e * adjacency_matrix), dim=1) 
        e_neighb = e_neighb - n_nodes + torch.tensor([self.g.out_degrees(i) for i in range(n_nodes)])
        e_neighb = torch.stack([e_neighb]*n_nodes, dim=1)

        alpha = e / e_neighb
        prod = torch.stack([h.unsqueeze(2)]*n_nodes,dim=2) * torch.stack([alpha.unsqueeze(1)]*n_nodes,dim=1)
        h2 = torch.sum(prod, dim=0).squeeze().T

        return self.activation(h2)


    def forward_attention(self, x, linear, attn, final_layer=False):
        n_nodes = x.shape[0]
        h = F.leaky_relu(linear(x), negative_slope=0.2)
        n_features = h.shape[1]

        adjacency_matrix = self.g.adjacency_matrix().coalesce().to(self.device)
        indices = adjacency_matrix.indices()
        values = []
        for i,j in zip(indices[0], indices[1]):
            values.append(attn(torch.cat((h[i],h[j]))))
        e = torch.sparse_coo_tensor(indices,values).to(self.device)

        alpha = torch.sparse.softmax(e, dim=1).coalesce()
        values = alpha.values()
        h2 = torch.zeros(n_nodes, n_features).to(self.device)
        for i,j,v in zip(indices[0], indices[1], values):
            h2[i,:] += h[j,:] * v

        if final_layer:
            #TODO
            pass

        return self.activation(h2)
       

    def add_simple_attention_layer(self, n_features, n_hidden_features):
        # TODO: multihead
        linear = Linear(n_features, n_hidden_features)
        attn = Linear(n_hidden_features*2, 1)

        self.linears.append(linear)
        self.attn_layers.append(attn)
        
