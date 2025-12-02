import torch
import  math

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
            torch.nn.SiLU(),
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


class PositionEmbedding(torch.nn.Module):
    def __init__(self, emb_dim):
        super(PositionEmbedding, self).__init__()
        self.emb_dim = emb_dim

    def forward(self, seq_len, device):
        position = torch.arange(seq_len).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, self.emb_dim, 2) * (-math.log(10000.0) / self.emb_dim)).to(device)
        trig_args = position * div_term
        pe = torch.zeros(seq_len, self.emb_dim).to(device)
        pe[:, 0::2] = torch.sin(trig_args)
        pe[:, 1::2] = torch.cos(trig_args)
        return pe

emb_dim=256
heads=32
num_layers=20
dict_size=60000
max_length=256
temperature=0.8


class MainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rms_norm = RMSNorm(emb_dim)
        self.embeddings = torch.nn.Embedding(num_embeddings=1114112, embedding_dim=emb_dim)
        self.pos_emb = PositionEmbedding(emb_dim)
        self.transformers = torch.nn.ModuleList([TransformerDecoder(emb_dim, heads) for _ in range(num_layers)])
        self.output_layer = torch.nn.Linear(emb_dim, dict_size)


    def forward(self, autoregressive, prompt):

        autoregressive_embed = self.embeddings(autoregressive)
        prompt_embed = self.embeddings(prompt)

        seq_len_prompt = prompt.shape[0]
        prompt_pos_enc = self.pos_emb(seq_len_prompt, autoregressive_embed.device).float()
        prompt_embed += prompt_pos_enc

        autoregressive_pos_enc = self.pos_emb(seq_len_prompt + 1, autoregressive_embed.device)[-1, :].unsqueeze(0).float()
        autoregressive_embed += autoregressive_pos_enc

        for block in self.transformers:
            autoregressive = block(autoregressive_embed, prompt_embed)

        autoregressive = self.rms_norm(autoregressive)
        autoregressive = torch.flatten(autoregressive)
        autoregressive = self.output_layer(autoregressive)
        return autoregressive
