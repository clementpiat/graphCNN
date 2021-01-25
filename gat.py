import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch import nn

class AttentionGraphModel(nn.Module):

    def __init__(self, g, n_layers, input_size, hidden_size, output_size, nonlinearity):
        """
        Highly inspired from https://arxiv.org/pdf/1710.10903.pdf
        """
        super().__init__()

        self.g = g
        self.adjacency_matrix = self.g.adjacency_matrix()
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

    def forward_attention(self, x, linear, attn, final_layer=False):
        # TODO: test that
        n_nodes = x.shape[0]
        h = F.leaky_relu(linear(x), negative_slope=0.2)
        e = torch.zeros((n_nodes,n_nodes))

        big_h_left = torch.stack([torch.stack([h[i]]*n_nodes, dim=0) for i in range(n_nodes)])
        big_h_right = torch.stack([h]*n_nodes, dim=0)
        big_h = torch.stack([big_h_left, big_h_right], dim=1)

        e = attn(big_h).view(n_nodes, n_nodes)

        
        e_neighb = torch.sum(torch.exp(e * self.adjacency_matrix), dim=1) 
        e_neighb = e_neighb - n_nodes + torch.tensor([self.g.out_degrees(i) for i in range(n_nodes)])
        e_neighb = torch.stack([e_neighb]*n_nodes, dim=1)

        alpha = torch.exp(e) / e_neighb * self.adjacency_matrix
        prod = torch.stack([h.unsqueeze(2)]*n_nodes,dim=2) * torch.stack([alpha.unsqueeze(1)]*n_nodes,dim=1)
        h2 = torch.sum(prod, dim=0).squeeze().T

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
        
