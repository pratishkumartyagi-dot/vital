"""Data processing utilities for PPG signal training."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


@dataclass
class PPGSignalData:
    """Container for PPG signal data."""
    red: np.ndarray
    green: np.ndarray
    blue: np.ndarray
    sampling_rate: int
    metadata: Dict[str, Any]
    vitals: Dict[str, float]
    ground_truth: Dict[str, float]
    quality: Dict[str, Any]


class PPGSignalProcessor:
    """Process raw PPG signals for training."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fs = config.get('sampling_rate', 30)
        self.window_size = int(config.get('window_size_seconds', 10) * self.fs)
        self.overlap = config.get('window_overlap', 0.5)
        self.bandpass_low = config.get('bandpass_low', 0.5)
        self.bandpass_high = config.get('bandpass_high', 4.0)
        
        # Design bandpass filter
        self._design_filter()
    
    def _design_filter(self):
        """Design bandpass filter for PPG signals."""
        nyquist = self.fs / 2
        low = self.bandpass_low / nyquist
        high = self.bandpass_high / nyquist
        self.b, self.a = scipy_signal.butter(4, [low, high], btype='band')
    
    def load_json(self, filepath: Path) -> PPGSignalData:
        """Load PPG signal from JSON export."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        signals = data['signals']
        metadata = data.get('metadata', {})
        vitals = data.get('vitals', {})
        ground_truth = data.get('groundTruth', {})
        quality = data.get('quality', {})
        
        return PPGSignalData(
            red=np.array(signals['red'], dtype=np.float32),
            green=np.array(signals['green'], dtype=np.float32),
            blue=np.array(signals['blue'], dtype=np.float32),
            sampling_rate=metadata.get('samplingRateHz', self.fs),
            metadata=metadata,
            vitals=vitals,
            ground_truth=ground_truth,
            quality=quality,
        )
    
    def bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to signal."""
        return scipy_signal.filtfilt(self.b, self.a, signal)
    
    def normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean, unit variance."""
        mean = np.mean(signal)
        std = np.std(signal)
        if std > 1e-6:
            return (signal - mean) / std
        return signal - mean
    
    def segment_signal(self, signal: np.ndarray) -> List[np.ndarray]:
        """Segment signal into overlapping windows."""
        step = int(self.window_size * (1 - self.overlap))
        windows = []
        for i in range(0, len(signal) - self.window_size + 1, step):
            window = signal[i:i + self.window_size]
            if len(window) == self.window_size:
                windows.append(window)
        return windows
    
    def extract_features(self, ppg_data: PPGSignalData) -> Dict[str, np.ndarray]:
        """Extract features from PPG signals."""
        # Bandpass filter each channel
        red_filt = self.bandpass_filter(ppg_data.red)
        green_filt = self.bandpass_filter(ppg_data.green)
        blue_filt = self.bandpass_filter(ppg_data.blue)
        
        # Normalize
        red_norm = self.normalize_signal(red_filt)
        green_norm = self.normalize_signal(green_filt)
        blue_norm = self.normalize_signal(blue_filt)
        
        # Stack channels: [3, T]
        multi_channel = np.stack([red_norm, green_norm, blue_norm], axis=0)
        
        # Segment into windows
        windows = []
        for ch in range(3):
            ch_windows = self.segment_signal(multi_channel[ch])
            windows.append(ch_windows)
        
        # Combine channels per window: [N_windows, 3, window_size]
        n_windows = len(windows[0])
        segmented = np.zeros((n_windows, 3, self.window_size), dtype=np.float32)
        for i in range(n_windows):
            for ch in range(3):
                segmented[i, ch] = windows[ch][i]
        
        return {
            'segmented': segmented,
            'filtered': multi_channel,
            'raw': np.stack([ppg_data.red, ppg_data.green, ppg_data.blue], axis=0),
        }
    
    def compute_targets(self, ppg_data: PPGSignalData) -> Dict[str, float]:
        """Compute target values from ground truth or app vitals."""
        gt = ppg_data.ground_truth
        vitals = ppg_data.vitals
        
        targets = {}
        
        # Heart rate
        targets['heart_rate'] = gt.get('bpm', vitals.get('heartRate', 0.0))
        
        # SpO2
        targets['spo2'] = gt.get('spo2', vitals.get('spo2', 0.0))
        
        # Hemoglobin
        targets['hemoglobin'] = gt.get('hb', vitals.get('hemoglobin', 0.0))
        
        # Respiratory rate
        targets['resp_rate'] = gt.get('respRate', vitals.get('respiratoryRate', 0.0))
        
        # Blood pressure (if available)
        targets['bp_sys'] = gt.get('bpSys', 0.0)
        targets['bp_dia'] = gt.get('bpDia', 0.0)
        
        # Glucose (if available)
        targets['glucose'] = gt.get('glucose', 0.0)
        
        return targets


class PPGDataset:
    """Dataset class for PPG training data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processor = PPGSignalProcessor(config)
        self.data_dir = Path(config.get('raw_signals_dir', './data/raw_signals'))
        self.samples = []
        self.targets = []
    
    def load_all(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Load all JSON files from data directory."""
        json_files = list(self.data_dir.glob('*.json'))
        logger.info(f"Found {len(json_files)} signal files")
        
        all_segments = []
        all_targets = {key: [] for key in ['heart_rate', 'spo2', 'hemoglobin', 'resp_rate', 'bp_sys', 'bp_dia', 'glucose']}
        
        for json_file in json_files:
            try:
                ppg_data = self.processor.load_json(json_file)
                
                # Skip if signal too short
                if len(ppg_data.green) < self.processor.window_size:
                    logger.warning(f"Skipping {json_file.name}: signal too short")
                    continue
                
                # Check signal quality
                pi = ppg_data.quality.get('perfusionIndex', 0)
                if pi < 0.05:  # Very low quality
                    logger.warning(f"Skipping {json_file.name}: low perfusion index {pi}")
                    continue
                
                # Extract features
                features = self.processor.extract_features(ppg_data)
                targets = self.processor.compute_targets(ppg_data)
                
                # Add each window as a sample
                segmented = features['segmented']
                for i in range(len(segmented)):
                    all_segments.append(segmented[i])
                    for key in all_targets:
                        all_targets[key].append(targets.get(key, 0.0))
                
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                continue
        
        # Convert to arrays
        X = np.array(all_segments, dtype=np.float32)  # [N, 3, T]
        y = {key: np.array(val, dtype=np.float32) for key, val in all_targets.items()}
        
        logger.info(f"Loaded {len(X)} samples")
        return X, y
    
    def split_data(self, X: np.ndarray, y: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Split data into train/val/test sets."""
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        
        train_split = self.config.get('train_split', 0.7)
        val_split = self.config.get('val_split', 0.15)
        
        train_end = int(n_samples * train_split)
        val_end = train_end + int(n_samples * val_split)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        splits = {}
        for split_name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            splits[split_name] = {
                'X': X[idx],
                'y': {key: val[idx] for key, val in y.items()},
            }
        
        return splits


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML."""
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)