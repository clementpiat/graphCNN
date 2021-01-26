import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch import nn
from tqdm import tqdm

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

    def forward(self, inputs):
        outputs = inputs
        for i, (linear, attn_layer) in enumerate(zip(self.linears, self.attn_layers)):
            outputs = self.forward_attention(outputs, linear, attn_layer, final_layer=(i==self.n_layers))
        return outputs

    # def forward_attention_dense(self, x, linear, attn, final_layer=False):
    #     n_nodes = x.shape[0]
    #     h = F.leaky_relu(linear(x), negative_slope=0.2)    
    #     adjacency_matrix = self.g.adjacency_matrix().coalesce()

    #     big_h_left = torch.stack([torch.stack([h[i]]*n_nodes, dim=0) for i in range(n_nodes)])
    #     big_h_right = torch.stack([h]*n_nodes, dim=0)
    #     big_h = torch.stack([big_h_left, big_h_right], dim=1)
    #     e = attn(big_h).view(n_nodes, n_nodes)

    #     e_neighb = torch.sum(torch.exp(e * adjacency_matrix), dim=1) 
    #     e_neighb = e_neighb - n_nodes + torch.tensor([self.g.out_degrees(i) for i in range(n_nodes)])
    #     e_neighb = torch.stack([e_neighb]*n_nodes, dim=1)

    #     alpha = e / e_neighb
    #     prod = torch.stack([h.unsqueeze(2)]*n_nodes,dim=2) * torch.stack([alpha.unsqueeze(1)]*n_nodes,dim=1)
    #     h2 = torch.sum(prod, dim=0).squeeze().T

    #     return self.activation(h2)

    def get_h2(self, attn_layer, h, indices):
        n_nodes, n_features = h.shape
        values = torch.zeros(len(indices[0]), self.n_head, device=self.device)
        h_conv = h.repeat(1,self.n_head).view(n_nodes,1,n_features*self.n_head,1)
        for k, (i,j) in enumerate(zip(indices[0], indices[1])):
            values[k,:] = attn_layer(torch.cat((h_conv[i],h_conv[j]), dim=1)).view(-1)

        e = torch.sparse_coo_tensor(indices,values)

        alpha = torch.sparse.softmax(e, dim=1).coalesce()
        values = alpha.values()
        h2 = torch.zeros(n_nodes, n_features, self.n_head, device=self.device)
        for i,j,v in zip(indices[0], indices[1], values):
            h2[i,:,:] += h[j,:].unsqueeze(-1) @ v.unsqueeze(0)

        return h2

    def get_h2_old(self, attn_layer, h, indices):
        n_nodes, n_features = h.shape
        values = torch.zeros(len(indices[0]), self.n_head, device=self.device)
        for k, (i,j) in enumerate(zip(indices[0], indices[1])):
            for p, attn_head in enumerate(attn_layer):
                values[k,p] = attn_head(torch.cat((h[i],h[j]))) 
        e = torch.sparse_coo_tensor(indices,values)

        alpha = torch.sparse.softmax(e, dim=1).coalesce()
        values = alpha.values()
        h2 = torch.zeros(n_nodes, n_features, self.n_head, device=self.device)
        for i,j,v in zip(indices[0], indices[1], values):
            h2[i,:,:] += h[j,:].unsqueeze(-1) @ v.unsqueeze(0)

        return h2

    def get_h2_old(self, attn, h, indices):
        values = torch.zeros(len(indices[0]), device=self.device)
        for k, (i,j) in enumerate(zip(indices[0], indices[1])):
            values[k] = attn(torch.cat((h[i],h[j])))
        e = torch.sparse_coo_tensor(indices,values)

        alpha = torch.sparse.softmax(e, dim=1).coalesce()
        values = alpha.values()
        h2 = torch.zeros(h.shape, device=self.device)
        for i,j,v in zip(indices[0], indices[1], values):
            h2[i,:] += h[j,:] * v

        return h2

    def forward_attention(self, x, linear, attn_layer, final_layer=False):
        h = F.leaky_relu(linear(x), negative_slope=0.2)

        adjacency_matrix = self.g.adjacency_matrix().coalesce().to(self.device)
        indices = adjacency_matrix.indices()
        # h2_list = [self.get_h2(attn_head, h, indices) for attn_head in attn_layer]

        # if final_layer:
        #     h2_list = list(map(lambda t: t.unsqueeze(0), h2_list))
        #     return self.activation(torch.cat(h2_list).mean(dim=0))
        # else:
        #     h2_list = list(map(self.activation, h2_list))
        #     return torch.cat(h2_list, dim=1)
        h2 = self.get_h2(attn_layer, h, indices)
        """
        h2 is of size (n_nodes,n_hidden_features,n_head)
        """
        if final_layer:
            return self.activation(h2.mean(dim=-1))
        else:
            return self.activation(h2.view(h2.size(0),-1))

    def add_attention_layer_old(self, n_features, n_hidden_features, n_head):
        linear = Linear(n_features, n_hidden_features)
        attn_layer = nn.ModuleList(
            [Linear(n_hidden_features*2, 1) for i in range(n_head)]
        )

        self.linears.append(linear)
        self.attn_layers.append(attn_layer)

    def add_attention_layer(self, n_features, n_hidden_features, n_head):
        linear = Linear(n_features, n_hidden_features)
        attn_layer = Conv1d(n_hidden_features*2*n_head, n_head, kernel_size=1, groups=n_head)
        
        self.linears.append(linear)
        self.attn_layers.append(attn_layer)
