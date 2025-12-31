import torch


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
        self.rope = RoPE(emb_size)  # Add RoPE layer
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder block"""
        # Self-attention with context
        residual = x
        x = self.rms_norm1(x)
        
        # Apply RoPE to query and key before attention
        # For MultiheadAttention, input shape is (seq_len, batch_size, emb_dim)
        # We need to apply RoPE to x (query) and context (key)
        x_rope = self.rope(x)
        context_rope = self.rope(context)
        
        x, _ = self.attention(x_rope, context_rope, context)
        x += residual
        
        # Feed forward network
        residual = x
        x = self.rms_norm2(x)
        x = self.feed_forward(x)
        x += residual
        
        return x

class RoPE(torch.nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, emb_dim: int, base: int = 10000):
        super().__init__()
        self.emb_dim = emb_dim
        self.base = base
        # Compute theta values
        self.theta = 1.0 / (base ** (torch.arange(0, emb_dim, 2) / emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to input tensor x of shape [seq_len, emb_dim] or [seq_len, batch_size, emb_dim]"""
        device = x.device
        
        # Handle both with and without batch dimension
        if len(x.shape) == 2:  # [seq_len, emb_dim]
            seq_len, emb_dim = x.shape
            batch_size = 1
            x = x.unsqueeze(1)  # Add batch dimension: [seq_len, 1, emb_dim]
        elif len(x.shape) == 3:  # [seq_len, batch_size, emb_dim]
            seq_len, batch_size, emb_dim = x.shape
        else:
            raise ValueError(f"Expected input shape [seq_len, emb_dim] or [seq_len, batch_size, emb_dim], got {x.shape}")
        
        # Expand theta to match sequence length
        theta = self.theta.to(device).unsqueeze(0).expand(seq_len, -1)
        
        # Create position tensor
        positions = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        
        # Compute cos and sin values
        cos_values = torch.cos(positions * theta).unsqueeze(1)
        sin_values = torch.sin(positions * theta).unsqueeze(1)
        
        # Reshape x to apply RoPE
        x_reshaped = x.view(seq_len, batch_size, -1, 2)
        
        # Apply rotation
        x_rotated = torch.stack([
            x_reshaped[..., 0] * cos_values - x_reshaped[..., 1] * sin_values,
            x_reshaped[..., 0] * sin_values + x_reshaped[..., 1] * cos_values
        ], dim=-1)
        
        # Remove batch dimension if it was added
        if batch_size == 1 and len(x_rotated.shape) == 4:
            return x_rotated.view(seq_len, emb_dim)
        else:
            return x_rotated.view(seq_len, batch_size, emb_dim)

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