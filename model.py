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
    """Transformer Decoder Block with RoPE"""
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
        self.rope = RotaryPositionEmbedding(emb_size)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder block"""
        # Self-attention with context
        residual = x
        x = self.rms_norm1(x)
        
        # Apply RoPE to query and key
        x_rope = self.rope(x.unsqueeze(0)).squeeze(0)
        context_rope = self.rope(context.unsqueeze(0)).squeeze(0)
        
        x, _ = self.attention(x_rope, context_rope, context)
        x += residual
        
        # Feed forward network
        residual = x
        x = self.rms_norm2(x)
        x = self.feed_forward(x)
        x += residual
        
        return x

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the dimensions of the tensor"""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class RotaryPositionEmbedding(torch.nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding to input tensor"""
        batch_size, seq_len, dim = x.shape
        assert dim == self.dim
        
        # Compute positions
        positions = torch.arange(0, seq_len, device=x.device).float()
        # Compute frequency-based positions
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq.to(x.device))
        # Duplicate frequencies to match the dimension
        emb = torch.cat((freqs, freqs), dim=-1)
        # Compute sine and cosine embeddings
        cos_emb = emb.cos().unsqueeze(0).expand(batch_size, -1, -1)
        sin_emb = emb.sin().unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply rotation to x
        x_rotated = (x * cos_emb) + (rotate_half(x) * sin_emb)
        
        return x_rotated

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
    """Main Transformer Model for Chatbot with RoPE"""
    def __init__(self):
        super().__init__()
        self.rms_norm = RMSNorm(CONFIG['emb_dim'])
        self.embeddings = torch.nn.Embedding(
            num_embeddings=CONFIG['dict_size'],  # Unified with dict_size
            embedding_dim=CONFIG['emb_dim']
        )
        self.transformers = torch.nn.ModuleList([
            TransformerDecoder(CONFIG['emb_dim'], CONFIG['heads'], CONFIG['dropout'])
            for _ in range(CONFIG['num_layers'])
        ])
        self.output_layer = torch.nn.Linear(CONFIG['emb_dim'], CONFIG['dict_size'])

    def forward(self, autoregressive: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model"""
        autoregressive_embed = self.embeddings(autoregressive)
        prompt_embed = self.embeddings(prompt)

        # Pass through transformer blocks
        x = autoregressive_embed
        for block in self.transformers:
            x = block(x, prompt_embed)

        # Output layer
        x = self.rms_norm(x)
        x = torch.flatten(x)
        x = self.output_layer(x)
        
        return x