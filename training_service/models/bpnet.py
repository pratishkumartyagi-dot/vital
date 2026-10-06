"""BPNet model architecture for PPG vital signs estimation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List


class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class BPNet(nn.Module):
    """
    BPNet: Blood Pressure Network for PPG-based vital signs estimation.
    
    Multi-task architecture predicting:
    - Heart Rate (regression)
    - SpO2 (regression)
    - Hemoglobin (regression)
    - Respiratory Rate (regression)
    - Blood Pressure Systolic/Diastolic (regression)
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_tasks: int = 6,  # HR, SpO2, Hb, RR, BP_sys, BP_dia
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
        # Initial convolution
        self.initial_conv = nn.Conv1d(input_channels, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.initial_bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks with increasing dilation
        self.res_blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.res_blocks.append(
                ResidualBlock(hidden_dim, kernel_size, dilation, dropout)
            )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            'heart_rate': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            ),
            'spo2': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            ),
            'hemoglobin': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            ),
            'resp_rate': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            ),
            'bp_sys': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            ),
            'bp_dia': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            ),
        })
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
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
        # Initial conv
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.relu(x)
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)  # [B, hidden_dim]
        
        # Task-specific predictions
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(x).squeeze(-1)
        
        return outputs
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
        }


class MultiTaskLoss(nn.Module):
    """Multi-task loss with uncertainty weighting (Kendall et al. 2018)."""
    
    def __init__(self, tasks: List[str], init_noise: float = 1.0):
        super().__init__()
        self.tasks = tasks
        # Learnable log variance for each task
        self.log_vars = nn.Parameter(torch.full((len(tasks),), init_noise))
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0
        losses = {}
        
        for i, task in enumerate(self.tasks):
            pred = predictions[task]
            target = targets[task]
            
            # MSE loss
            mse = F.mse_loss(pred, target)
            
            # Uncertainty weighting
            precision = torch.exp(-self.log_vars[i])
            loss = precision * mse + self.log_vars[i]
            
            losses[task] = mse.detach()
            total_loss += loss
        
        return total_loss, losses


def create_bpnet(config: Dict[str, Any]) -> BPNet:
    """Factory function to create BPNet from config."""
    model_config = config.get('models', {}).get('bpnet', {})
    return BPNet(
        input_channels=model_config.get('input_channels', 3),
        hidden_dim=model_config.get('hidden_dim', 64),
        num_layers=model_config.get('num_layers', 4),
        kernel_size=model_config.get('kernel_size', 3),
        dropout=model_config.get('dropout', 0.2),
    )