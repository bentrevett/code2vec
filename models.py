import torch
import torch.nn as nn
import torch.nn.functional as F

class Code2Vec(nn.Module):
    def __init__(self, nodes_dim, paths_dim, embedding_dim, output_dim, dropout):
        super().__init__()
        
        self.node_embedding = nn.Embedding(nodes_dim, embedding_dim)
        self.path_embedding = nn.Embedding(paths_dim, embedding_dim)
        self.W = nn.Parameter(torch.randn(1, embedding_dim, 3*embedding_dim))
        self.a = nn.Parameter(torch.randn(1, embedding_dim, 1))
        self.out = nn.Linear(embedding_dim, output_dim)
        self.do = nn.Dropout(dropout)
        
    def forward(self, starts, paths, ends):
        
        #starts = paths = ends = [batch size, max length]
        
        W = self.W.repeat(starts.shape[0], 1, 1)
        
        #W = [batch size, embedding dim, embedding dim * 3]
        
        embedded_starts = self.node_embedding(starts)
        embedded_paths = self.path_embedding(paths)
        embedded_ends = self.node_embedding(ends)
        
        #embedded_* = [batch size, max length, embedding dim]

        c = self.do(torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2))
        
        #c = [batch size, max length, embedding dim * 3]
        
        c = c.permute(0, 2, 1)
        
        #c = [batch size, embedding dim * 3, max length]

        x = torch.tanh(torch.bmm(W, c))
        
        #x = [batch size, embedding dim, max length]
        
        x = x.permute(0, 2, 1)
        
        #x = [batch size, max length, embedding dim]
        
        a = self.a.repeat(starts.shape[0], 1, 1)
        
        #a = [batch size, embedding dim, 1]

        z = torch.bmm(x, a).squeeze(2)
        
        #z = [batch size, max length]

        z = F.softmax(z, dim=1)
        
        #z = [batch size, max length]
        
        z = z.unsqueeze(2)
        
        #z = [batch size, max length, 1]
        
        x = x.permute(0, 2, 1)
        
        #x = [batch size, embedding dim, max length]
        
        v = torch.bmm(x, z).squeeze(2)
        
        #v = [batch size, embedding dim]
        
        out = self.out(v)
        
        #out = [batch size, output dim]

        return out