"""PhysFormer model architecture for PPG vital signs estimation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import math


class PatchEmbedding(nn.Module):
    """Patch embedding for 1D signals."""
    
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, embed_dim, T/patch_size] -> [B, T/patch_size, embed_dim]
        x = self.projection(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """MLP block."""
    
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PhysFormer(nn.Module):
    """
    PhysFormer: Transformer-based PPG vital signs estimation.
    
    Based on "PhysFormer: A Transformer-based Model for Physiological Signal Analysis"
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        patch_size: int = 16,
        dropout: float = 0.1,
        num_tasks: int = 6,
        max_seq_len: int = 192,  # window_size / patch_size = 300/16 ≈ 19
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(input_channels, embed_dim, patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            'heart_rate': nn.Linear(embed_dim, 1),
            'spo2': nn.Linear(embed_dim, 1),
            'hemoglobin': nn.Linear(embed_dim, 1),
            'resp_rate': nn.Linear(embed_dim, 1),
            'bp_sys': nn.Linear(embed_dim, 1),
            'bp_dia': nn.Linear(embed_dim, 1),
        })
        
        # Class token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, C, T] where C=3 (RGB), T=time steps
            
        Returns:
            Dictionary of predictions for each task
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, embed_dim]
        N = x.shape[1]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :N+1]
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Use class token for global representation
        cls_output = x[:, 0]  # [B, embed_dim]
        
        # Task-specific predictions
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(cls_output).squeeze(-1)
        
        return outputs
    
    def get_model_size(self) -> Dict[str, int]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
        }


def create_physformer(config: Dict[str, Any]) -> PhysFormer:
    """Factory function to create PhysFormer from config."""
    model_config = config.get('models', {}).get('physformer', {})
    return PhysFormer(
        input_channels=model_config.get('input_channels', 3),
        embed_dim=model_config.get('embed_dim', 128),
        num_heads=model_config.get('num_heads', 8),
        num_layers=model_config.get('num_layers', 6),
        patch_size=model_config.get('patch_size', 16),
        dropout=model_config.get('dropout', 0.1),
    )