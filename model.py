import torch
import math
from typing import Tuple

class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class TransformerDecoder(torch.nn.Module):
    """Transformer Decoder Block"""
    def __init__(self, emb_size: int, num_heads: int, drop_out: float = 0.1):
        super().__init__()
        self.rms_norm1 = RMSNorm(emb_size)
        self.attention = torch.nn.MultiheadAttention(emb_size, num_heads, dropout=drop_out)
        self.rms_norm2 = RMSNorm(emb_size)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(emb_size, 4 * emb_size),
            torch.nn.Dropout(drop_out),
            torch.nn.SiLU(),
            torch.nn.Linear(4 * emb_size, emb_size),
            torch.nn.Dropout(drop_out)
        )
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder block"""
        # Self-attention with context
        residual = x
        x = self.rms_norm1(x)
        x, _ = self.attention(x, context, context)
        x += residual
        
        # Feed forward network
        residual = x
        x = self.rms_norm2(x)
        x = self.feed_forward(x)
        x += residual
        
        return x

class PositionEmbedding(torch.nn.Module):
    """Positional Encoding"""
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate positional encoding for a sequence of length seq_len"""
        position = torch.arange(seq_len).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, self.emb_dim, 2) * (-math.log(10000.0) / self.emb_dim)).to(device)
        trig_args = position * div_term
        
        pe = torch.zeros(seq_len, self.emb_dim).to(device)
        pe[:, 0::2] = torch.sin(trig_args)
        pe[:, 1::2] = torch.cos(trig_args)
        
        return pe

# Model configuration
CONFIG = {
    'emb_dim': 256,
    'heads': 8,  # Reduced from 32 for better performance
    'num_layers': 6,  # Reduced from 20 for better performance
    'dict_size': 60000,
    'max_length': 256,
    'temperature': 0.8,
    'dropout': 0.1
}

class MainModel(torch.nn.Module):
    """Main Transformer Model for Chatbot"""
    def __init__(self):
        super().__init__()
        self.rms_norm = RMSNorm(CONFIG['emb_dim'])
        self.embeddings = torch.nn.Embedding(
            num_embeddings=CONFIG['dict_size'],  # Unified with dict_size
            embedding_dim=CONFIG['emb_dim']
        )
        self.pos_emb = PositionEmbedding(CONFIG['emb_dim'])
        self.transformers = torch.nn.ModuleList([
            TransformerDecoder(CONFIG['emb_dim'], CONFIG['heads'], CONFIG['dropout'])
            for _ in range(CONFIG['num_layers'])
        ])
        self.output_layer = torch.nn.Linear(CONFIG['emb_dim'], CONFIG['dict_size'])

    def forward(self, autoregressive: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model"""
        autoregressive_embed = self.embeddings(autoregressive)
        prompt_embed = self.embeddings(prompt)

        # Add positional encoding
        seq_len_prompt = prompt.shape[0]
        prompt_pos_enc = self.pos_emb(seq_len_prompt, autoregressive_embed.device).float()
        prompt_embed += prompt_pos_enc

        # For autoregressive input, we only need the position encoding for its position
        autoregressive_pos_enc = self.pos_emb(seq_len_prompt + 1, autoregressive_embed.device)[-1, :].unsqueeze(0).float()
        autoregressive_embed += autoregressive_pos_enc

        # Pass through transformer blocks
        x = autoregressive_embed
        for block in self.transformers:
            x = block(x, prompt_embed)

        # Output layer
        x = self.rms_norm(x)
        x = torch.flatten(x)
        x = self.output_layer(x)
        
        return x