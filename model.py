import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class TransformerDecoder(torch.nn.Module):
    def __init__(self,emb_size,num_heads,drop_out=0.1):
        super().__init__()
        self.rms_norm=RMSNorm(emb_size)
        self.attention=torch.nn.MultiheadAttention(emb_size,num_heads,dropout=drop_out)
        self.feed_forward=torch.nn.Sequential(
            torch.nn.Linear(emb_size,4*emb_size),
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(4*emb_size,emb_size),
            torch.nn.Dropout(drop_out)
        )
    def forward(self,autoregressive,prompt):
        residuals=autoregressive
        autoregressive=self.rms_norm(autoregressive)
        autoregressive,_=self.attention(autoregressive,prompt,prompt)
        autoregressive+=residuals
        residuals = autoregressive
        autoregressive=self.rms_norm(autoregressive)
        autoregressive=self.feed_forward(autoregressive)
        autoregressive+=residuals
        return autoregressive

emb_dim=256
heads=32
num_layers=12
dict_size=60000
max_length=256
temperature=0.8

class MainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings=torch.nn.Embedding(num_embeddings=1114112,embedding_dim=emb_dim)
        self.transformers=torch.nn.ModuleList([TransformerDecoder(emb_dim,heads) for i in range(num_layers)])
        self.output_layer=torch.nn.Linear(emb_dim,dict_size)
    def forward(self,autoregressive,prompt):
        autoregressive=self.embeddings(autoregressive)
        prompt=self.embeddings(prompt)
        for block in self.transformers:
            autoregressive=block(autoregressive,prompt)
        autoregressive=torch.flatten(autoregressive)
        autoregressive=self.output_layer(autoregressive)
        return autoregressive
