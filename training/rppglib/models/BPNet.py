import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.context_vector = nn.Parameter(torch.randn(hidden_dim))
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out):
        scores = self.attention(lstm_out).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)
        return context, weights


class BPNet(nn.Module):
    """
    CNN-BiLSTM-Attention model for cuff-less BP estimation from PPG signals.
    Based on: Mohammadi et al., "Cuff-less blood pressure monitoring via PPG signals
    using a hybrid CNN-BiLSTM deep learning model with attention mechanism", Sci Rep 2025.
    
    Architecture:
    - 3 CNN layers: filters=[32, 64, 128], kernel_size=3, stride=1, maxpool=2
    - 2 BiLSTM layers: hidden_size=128, bidirectional=True
    - Attention layer: context vector attention
    - Output: SBP and DBP regression heads
    """
    def __init__(
        self,
        input_length=1024,
        cnn_filters=[32, 64, 128],
        kernel_size=3,
        pool_size=2,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.2,
        num_outputs=2
    ):
        super().__init__()
        
        self.input_length = input_length
        self.cnn_filters = cnn_filters
        self.lstm_hidden = lstm_hidden
        
        # CNN layers
        self.cnn = nn.Sequential()
        in_channels = 1
        current_length = input_length
        
        for i, out_channels in enumerate(cnn_filters):
            self.cnn.add_module(f'conv{i}', nn.Conv1d(
                in_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2
            ))
            self.cnn.add_module(f'bn{i}', nn.BatchNorm1d(out_channels))
            self.cnn.add_module(f'relu{i}', nn.ReLU())
            self.cnn.add_module(f'pool{i}', nn.MaxPool1d(pool_size, stride=pool_size))
            self.cnn.add_module(f'dropout{i}', nn.Dropout(dropout))
            in_channels = out_channels
            current_length = current_length // pool_size
        
        self.cnn_output_length = current_length
        self.cnn_output_channels = cnn_filters[-1]
        
        # BiLSTM layers
        self.bilstm = nn.LSTM(
            input_size=self.cnn_output_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = AttentionLayer(lstm_hidden * 2)
        
        # Output heads for SBP and DBP
        self.sbp_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.dbp_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x: (batch, 1, input_length) or (batch, input_length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # CNN feature extraction
        x = self.cnn(x)  # (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)
        
        # BiLSTM temporal modeling
        lstm_out, _ = self.bilstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention
        context, weights = self.attention(lstm_out)  # (batch, hidden*2)
        
        # Regression heads
        sbp = self.sbp_head(context).squeeze(-1)
        dbp = self.dbp_head(context).squeeze(-1)
        
        return sbp, dbp, weights


class BPNetTrainer:
    def __init__(
        self,
        model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lr=0.001,
        weight_decay=1e-4
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.criterion = nn.MSELoss()
        
    def train_step(self, ppg, sbp_target, dbp_target):
        self.model.train()
        ppg = ppg.to(self.device)
        sbp_target = sbp_target.to(self.device)
        dbp_target = dbp_target.to(self.device)
        
        self.optimizer.zero_grad()
        sbp_pred, dbp_pred, _ = self.model(ppg)
        
        loss_sbp = self.criterion(sbp_pred, sbp_target)
        loss_dbp = self.criterion(dbp_pred, dbp_target)
        loss = loss_sbp + loss_dbp
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'loss_sbp': loss_sbp.item(),
            'loss_dbp': loss_dbp.item(),
            'mae_sbp': F.l1_loss(sbp_pred, sbp_target).item(),
            'mae_dbp': F.l1_loss(dbp_pred, dbp_target).item()
        }
    
    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_mae_sbp = 0
        total_mae_dbp = 0
        count = 0
        
        for ppg, sbp_target, dbp_target in val_loader:
            ppg = ppg.to(self.device)
            sbp_target = sbp_target.to(self.device)
            dbp_target = dbp_target.to(self.device)
            
            sbp_pred, dbp_pred, _ = self.model(ppg)
            
            loss_sbp = self.criterion(sbp_pred, sbp_target)
            loss_dbp = self.criterion(dbp_pred, dbp_target)
            total_loss += (loss_sbp + loss_dbp).item()
            total_mae_sbp += F.l1_loss(sbp_pred, sbp_target).item()
            total_mae_dbp += F.l1_loss(dbp_pred, dbp_target).item()
            count += 1
        
        return {
            'val_loss': total_loss / count,
            'val_mae_sbp': total_mae_sbp / count,
            'val_mae_dbp': total_mae_dbp / count
        }
    
    def save_checkpoint(self, path, epoch, metrics):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']


def create_bpnet(config=None):
    """Factory function to create BPNet with paper's optimal hyperparameters."""
    default_config = {
        'input_length': 1024,      # 8.192s @ 125Hz
        'cnn_filters': [32, 64, 128],
        'kernel_size': 3,
        'pool_size': 2,
        'lstm_hidden': 128,
        'lstm_layers': 2,
        'dropout': 0.2,
        'num_outputs': 2
    }
    if config:
        default_config.update(config)
    return BPNet(**default_config)