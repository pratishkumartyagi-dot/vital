"""
ROSE-Net: Remote Oxygen Saturation Estimation Network

Deep learning architecture for non-contact SpO2 estimation via rPPG from
facial video streams. Uses cross-domain training: learns core spatial-temporal
features from clinical contact PPG data (BIDMC dataset) and transfers that
knowledge to camera-based rPPG video streams.

Reference:
    Chowdhury, M.H., Reaz, M.B.I., Ali, S.H.Md., Khan, M.S., Chowdhury, M.E.H.
    "ROSE-Net: Leveraging remote photoplethysmography to estimate oxygen
    saturation using deep learning."
    Biomedical Signal Processing and Control, vol. 100, 2025, 107105.
    doi: 10.1016/j.bspc.2024.107105

STMap (Spatial-Temporal Map) input:
    A 2-D representation where one axis is time (frame index) and the other
    axis corresponds to colour channels (R, G, B).  Each pixel stores the
    mean-normalised channel intensity.  Additional colour spaces (YUV,
    YCrCb) can be concatenated to form a multi-channel image that is fed
    directly into a 2-D CNN backbone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from .base_model import BaseModel


class ROSENetBackbone(nn.Module):
    """CNN backbone that extracts spatio-temporal features from STMaps.

    Accepts an STMap of shape ``(batch, channels, height, width)`` where
    ``channels`` is the number of colour-space channels (e.g. 3 for RGB,
    9 for RGB+YUV+YCrCb), ``height`` is the channel axis (typically 1 for
    a single-row STMap) and ``width`` is the time axis.

    The architecture uses residual blocks with group convolutions to
    encourage separable feature learning across colour channels and time.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_blocks: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        blocks = []

        # Initial conv: expand spatial and temporal dimensions
        blocks.append(
            nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
            )
        )

        channels = base_channels
        for i in range(num_blocks):
            out_channels = channels * 2
            blocks.append(_ROSENetBlock(channels, out_channels))
            channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.dropout(x)
        # Global average pooling over both spatial dims
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


class _ROSENetBlock(nn.Module):
    """Residual block with channel expansion."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ROSENet(nn.Module):
    """ROSE-Net: Spatial-Temporal CNN for non-contact SpO2 estimation.

    Architecture:
        - STMap input (multi-channel 2-D image: colour channels × time)
        - Residual CNN backbone with channel doubling
        - Global average pooling
        - Temporal attention pooling (optional, for variable-length input)
        - FC regression head → SpO2 (%)

    Cross-domain training strategy:
        1. Pre-train on BIDMC contact-PPG-derived STMaps (source domain)
        2. Fine-tune on facial-video rPPG STMaps (target domain)

    Args:
        in_channels:  Number of colour-space channels in the STMap.
        input_length: Expected time-axis length of the STMap.
        base_channels: Initial CNN channel width.
        num_blocks:   Number of residual blocks.
        lstm_hidden:  Hidden size of the BiLSTM temporal modulator.
        dropout:      Dropout probability.
        use_attention: Whether to apply temporal self-attention after BiLSTM.
    """

    def __init__(
        self,
        in_channels: int = 3,
        input_length: int = 256,
        base_channels: int = 32,
        num_blocks: int = 4,
        lstm_hidden: int = 64,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_length = input_length
        self.use_attention = use_attention

        # --- CNN spatial-temporal feature extractor ---
        self.backbone = ROSENetBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            num_blocks=num_blocks,
            dropout=dropout,
        )

        feat_dim = self.backbone.out_channels

        # --- Temporal modelling: BiLSTM over pooled features ---
        # For variable-length sequences we run the CNN per-frame and
        # then aggregate with a BiLSTM + attention.
        self.temporal_conv = nn.Conv1d(feat_dim, lstm_hidden, kernel_size=1)
        self.temporal_bn = nn.BatchNorm1d(lstm_hidden)
        self.bilstm = nn.LSTM(
            lstm_hidden, lstm_hidden, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout if dropout > 0 else 0,
        )

        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_hidden * 2, lstm_hidden),
                nn.Tanh(),
                nn.Linear(lstm_hidden, 1),
            )
        else:
            self.attention = None

        # --- SpO2 regression head ---
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: STMap tensor of shape ``(batch, channels, 1, time)`` or
               ``(batch, channels, time)``.

        Returns:
            Predicted SpO2 values of shape ``(batch, 1)``.
        """
        # Ensure 4-D input: (B, C, H=1, W=time)
        if x.dim() == 3:
            x = x.unsqueeze(2)

        batch_size = x.size(0)

        # CNN feature extraction — output is (B, feat_dim) per time column
        # We process the whole STMap at once for fixed-length input.
        feats = self.backbone(x)  # (B, feat_dim)

        # For variable-length sequences we'd chunk; here we treat the
        # pooled features as a single time step.  When the input is long
        # enough, we can also split into segments for temporal modelling.
        if self.input_length >= 128 and x.size(-1) >= 128:
            # Segment-level temporal modelling
            time_dim = x.size(-1)
            seg_len = 32
            n_seg = max(1, time_dim // seg_len)

            seg_feats = []
            for s in range(n_seg):
                seg = x[:, :, :, s * seg_len:(s + 1) * seg_len]
                seg_feats.append(self.backbone(seg))

            seg_feats = torch.stack(seg_feats, dim=1)  # (B, n_seg, feat_dim)
            seg_feats = self.temporal_bn(self.temporal_conv(
                seg_feats.transpose(1, 2)
            ).transpose(1, 2))  # (B, n_seg, lstm_hidden)

            lstm_out, _ = self.bilstm(seg_feats)  # (B, n_seg, 2*lstm_hidden)

            if self.use_attention:
                scores = self.attention(lstm_out).squeeze(-1)  # (B, n_seg)
                weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, n_seg, 1)
                context = torch.sum(lstm_out * weights, dim=1)  # (B, 2*lstm_hidden)
            else:
                context = lstm_out.mean(dim=1)
        else:
            # Short input: use pooled features directly
            context = feats
            # Project to the BiLSTM output dimension
            context = F.relu(self.temporal_bn(
                self.temporal_conv(context.unsqueeze(-1))
            ).squeeze(-1))  # (B, lstm_hidden)
            context = torch.cat([context, context], dim=1)  # (B, 2*lstm_hidden)

        spo2 = self.regressor(context)  # (B, 1)
        return spo2


class ROSENetWrapper(BaseModel):
    """Training/inference wrapper following the project's BaseModel pattern."""

    def __init__(self, config):
        super().__init__(config)
        self.require_train = True

        self.net = ROSENet(
            in_channels=getattr(config, 'in_channels', 3),
            input_length=getattr(config, 'input_length', 256),
            base_channels=getattr(config, 'base_channels', 32),
            num_blocks=getattr(config, 'num_blocks', 4),
            lstm_hidden=getattr(config, 'lstm_hidden', 64),
            dropout=getattr(config, 'dropout', 0.3),
            use_attention=getattr(config, 'use_attention', True),
        )
        self.device = config.device
        self.weights_name = (
            f'{config.results_folder}/ROSENet__{config.train_dataset}__'
            f'{config.valid_fold}.pt'
        )

        if hasattr(config, 'pretrained_weights') and config.pretrained_weights:
            state_dict = torch.load(config.pretrained_weights, map_location='cpu')
            self.net.load_state_dict(state_dict, strict=False)

    def predict(self, stmaps):
        """Predict SpO2 from STMap tensors.

        Args:
            stmaps: Tensor of shape (B, C, 1, T) or (B, C, T).

        Returns:
            SpO2 predictions of shape (B, 1).
        """
        if isinstance(stmaps, np.ndarray):
            stmaps = torch.from_numpy(stmaps).float()
        self.net.cpu()
        self.net.eval()
        with torch.no_grad():
            preds = self.net(stmaps)
        return preds

    @staticmethod
    def stmap_processing(video, landmarks):
        """Generate STMap from video frames.

        This is a thin wrapper that delegates to ``rppglib.stmap``.
        Returns a numpy array of shape (T, C) representing the
        mean-normalised RGB values over time.
        """
        from ..stmap import generate_stmap
        return generate_stmap(video, landmarks)


def create_rosenet(config: Optional[dict] = None) -> ROSENet:
    """Factory function to create a ROSE-Net model with paper defaults."""
    defaults = {
        'in_channels': 3,
        'input_length': 256,
        'base_channels': 32,
        'num_blocks': 4,
        'lstm_hidden': 64,
        'dropout': 0.3,
        'use_attention': True,
    }
    if config:
        defaults.update(config)
    return ROSENet(**defaults)


if __name__ == '__main__':
    model = create_rosenet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'ROSE-Net created with {n_params:,} parameters')

    # Test forward pass
    batch_size = 4
    channels = 3
    time_steps = 256
    x = torch.randn(batch_size, channels, 1, time_steps)
    out = model(x)
    print(f'Input shape:  {x.shape}')
    print(f'Output shape: {out.shape}')
    print(f'Sample SpO2 predictions: {out.squeeze().tolist()}')
