"""
Hybrid CNN-BiLSTM with Attention Mechanism for Cuff-less BP Estimation from PPG
Based on: Mohammadi et al., "Cuff-less blood pressure monitoring via PPG signals 
using a hybrid CNN-BiLSTM deep learning model with attention mechanism" (2025)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentionLayer(nn.Module):
    """Attention mechanism for focusing on relevant temporal segments of PPG signal"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_vector = nn.Parameter(torch.randn(hidden_dim))
        self.attention = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: BiLSTM output of shape (batch, seq_len, hidden_dim*2)
        Returns:
            context_vector: Weighted sum of hidden states (batch, hidden_dim*2)
            attention_weights: Attention weights for each time step (batch, seq_len)
        """
        scores = self.attention(x).squeeze(-1)  # (batch, seq_len)
        attention_weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        
        # Weighted sum of hidden states
        context = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim*2)
        
        return context, attention_weights


class CNNFeatureExtractor(nn.Module):
    """3-layer CNN for spatial feature extraction from PPG signals"""
    
    def __init__(self, in_channels: int = 1, filters: list = [32, 64, 128], 
                 kernel_size: int = 3, pool_size: int = 2, dropout: float = 0.2):
        super().__init__()
        
        layers = []
        current_channels = in_channels
        
        for i, out_channels in enumerate(filters):
            layers.extend([
                nn.Conv1d(current_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(pool_size, stride=pool_size),
                nn.Dropout(dropout)
            ])
            current_channels = out_channels
            
        self.cnn = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input PPG signal (batch, 1, seq_len)
        Returns:
            Feature maps (batch, channels, seq_len_reduced)
        """
        return self.cnn(x)


class BiLSTMEncoder(nn.Module):
    """2-layer BiLSTM for temporal dependency capture"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CNN features (batch, seq_len, channels)
        Returns:
            BiLSTM output (batch, seq_len, hidden_dim * num_directions)
        """
        output, _ = self.lstm(x)
        return output


class BPNet(nn.Module):
    """
    Hybrid CNN-BiLSTM-Attention Network for BP Estimation from PPG
    
    Architecture (from paper):
    - 3 CNN layers with filters [32, 64, 128], kernel=3, pool=2, ReLU
    - 2 BiLSTM layers with 128 hidden units each
    - 1 Attention layer
    - Output: SBP and DBP regression heads
    """
    
    def __init__(self, 
                 input_length: int = 1024,
                 cnn_filters: list = [32, 64, 128],
                 kernel_size: int = 3,
                 pool_size: int = 2,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        
        self.input_length = input_length
        
        # Calculate CNN output length
        cnn_out_len = input_length
        for _ in cnn_filters:
            cnn_out_len = cnn_out_len // pool_size
            
        self.cnn = CNNFeatureExtractor(
            in_channels=1,
            filters=cnn_filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            dropout=dropout
        )
        
        # CNN output channels = last filter size
        cnn_out_channels = cnn_filters[-1]
        
        self.bilstm = BiLSTMEncoder(
            input_dim=cnn_out_channels,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = AttentionLayer(hidden_dim=lstm_hidden * 2)
        
        # Regression heads for SBP and DBP
        self.sbp_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.dbp_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: PPG signal (batch, 1, seq_len) or (batch, seq_len)
        Returns:
            sbp: Predicted systolic BP (batch, 1)
            dbp: Predicted diastolic BP (batch, 1)
            attention_weights: Attention weights for visualization (batch, seq_len)
        """
        # Ensure correct input shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, seq_len)
        elif x.dim() == 3 and x.shape[1] != 1:
            x = x.transpose(1, 2)  # (batch, seq_len, channels) -> (batch, channels, seq_len)
            
        # CNN feature extraction
        cnn_features = self.cnn(x)  # (batch, channels, seq_len_reduced)
        
        # Transpose for LSTM: (batch, seq_len, channels)
        lstm_input = cnn_features.transpose(1, 2)
        
        # BiLSTM temporal encoding
        bilstm_output = self.bilstm(lstm_input)  # (batch, seq_len, hidden*2)
        
        # Attention mechanism
        context, attention_weights = self.attention(bilstm_output)
        
        # Regression heads
        sbp = self.sbp_head(context)
        dbp = self.dbp_head(context)
        
        return sbp, dbp, attention_weights


def create_model(config: Optional[dict] = None) -> BPNet:
    """Factory function to create BPNet model"""
    default_config = {
        'input_length': 1024,
        'cnn_filters': [32, 64, 128],
        'kernel_size': 3,
        'pool_size': 2,
        'lstm_hidden': 128,
        'lstm_layers': 2,
        'dropout': 0.2
    }
    if config:
        default_config.update(config)
    return BPNet(**default_config)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Test forward pass
    batch_size = 4
    seq_len = 1024
    x = torch.randn(batch_size, 1, seq_len)
    
    sbp, dbp, attn = model(x)
    print(f"Input shape: {x.shape}")
    print(f"SBP output shape: {sbp.shape}")
    print(f"DBP output shape: {dbp.shape}")
    print(f"Attention weights shape: {attn.shape}")