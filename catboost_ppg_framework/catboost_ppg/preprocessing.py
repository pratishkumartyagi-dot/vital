"""
PPG Signal Preprocessing Module
================================

Implements the preprocessing pipeline for fingertip PPG signals:
- Bandpass filtering (0.5-16 Hz)
- Baseline correction / detrending
- Motion artifact detection and removal
- Min-Max normalization
- Signal quality assessment
"""

import numpy as np
import scipy.signal as signal
from scipy.signal import butter, filtfilt, find_peaks
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class PreprocessingConfig:
    """Configuration for PPG preprocessing pipeline."""
    fs: float = 125.0  # Sampling frequency (Hz)
    low_freq: float = 0.5  # Bandpass low cutoff (Hz)
    high_freq: float = 16.0  # Bandpass high cutoff (Hz)
    filter_order: int = 4
    detrend_order: int = 2  # Polynomial order for detrending
    min_signal_length: float = 10.0  # Minimum signal duration (seconds)
    max_signal_length: float = 120.0  # Maximum signal duration (seconds)
    motion_threshold: float = 3.0  # Z-score threshold for motion artifacts
    saturation_threshold: float = 0.99  # Fraction of max ADC for saturation detection


@dataclass
class SignalQualityMetrics:
    """Signal quality assessment results."""
    snr_db: float
    perfusion_index: float
    saturation_ratio: float
    motion_artifact_ratio: float
    overall_quality: float  # 0-100
    is_acceptable: bool
    quality_flags: Dict[str, bool]


class PPGPreprocessor:
    """
    Preprocessor for fingertip PPG signals.
    
    Handles the complete preprocessing pipeline including filtering,
    detrending, artifact removal, and normalization.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self._filter_cache: Dict[Tuple, Tuple] = {}
    
    def _get_filter_coeffs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get or create Butterworth bandpass filter coefficients."""
        key = (self.config.fs, self.config.low_freq, self.config.high_freq, self.config.filter_order)
        if key not in self._filter_cache:
            nyquist = self.config.fs / 2
            low = self.config.low_freq / nyquist
            high = self.config.high_freq / nyquist
            b, a = butter(self.config.filter_order, [low, high], btype='band')
            self._filter_cache[key] = (b, a)
        return self._filter_cache[key]
    
    def bandpass_filter(self, ppg: np.ndarray) -> np.ndarray:
        """Apply zero-phase Butterworth bandpass filter."""
        if len(ppg) < 10:
            return ppg
        b, a = self._get_filter_coeffs()
        try:
            filtered = filtfilt(b, a, ppg)
            return filtered.astype(np.float32)
        except Exception:
            return ppg
    
    def detrend_signal(self, ppg: np.ndarray) -> np.ndarray:
        """Remove baseline wander using polynomial detrending."""
        if len(ppg) < 5:
            return ppg
        x = np.arange(len(ppg))
        try:
            coeffs = np.polyfit(x, ppg, self.config.detrend_order)
            trend = np.polyval(coeffs, x)
            return ppg - trend
        except Exception:
            return ppg
    
    def normalize_minmax(self, ppg: np.ndarray) -> np.ndarray:
        """Min-Max normalization to [0, 1] range."""
        ppg_min, ppg_max = np.min(ppg), np.max(ppg)
        if ppg_max - ppg_min < 1e-8:
            return np.zeros_like(ppg)
        return (ppg - ppg_min) / (ppg_max - ppg_min)
    
    def normalize_zscore(self, ppg: np.ndarray) -> np.ndarray:
        """Z-score normalization (zero mean, unit variance)."""
        mean, std = np.mean(ppg), np.std(ppg)
        if std < 1e-8:
            return np.zeros_like(ppg)
        return (ppg - mean) / std
    
    def detect_saturation(self, ppg: np.ndarray, bit_depth: int = 12) -> float:
        """Detect signal saturation (clipping at ADC limits)."""
        max_val = 2 ** bit_depth - 1
        saturated = np.sum(ppg >= max_val * self.config.saturation_threshold)
        return saturated / len(ppg)
    
    def detect_motion_artifacts(self, ppg: np.ndarray) -> np.ndarray:
        """
        Detect motion artifacts using moving window statistics.
        Returns boolean mask where True indicates artifact.
        """
        window_size = int(self.config.fs * 1.0)  # 1-second windows
        if len(ppg) < window_size * 2:
            return np.zeros(len(ppg), dtype=bool)
        
        artifact_mask = np.zeros(len(ppg), dtype=bool)
        half_window = window_size // 2
        
        for i in range(half_window, len(ppg) - half_window):
            window = ppg[i - half_window:i + half_window]
            z_scores = np.abs((window - np.mean(window)) / (np.std(window) + 1e-8))
            if np.mean(z_scores > self.config.motion_threshold) > 0.3:
                artifact_mask[i] = True
        
        return artifact_mask
    
    def interpolate_artifacts(self, ppg: np.ndarray, artifact_mask: np.ndarray) -> np.ndarray:
        """Interpolate over detected motion artifacts."""
        if not np.any(artifact_mask):
            return ppg
        
        clean_ppg = ppg.copy()
        artifact_indices = np.where(artifact_mask)[0]
        
        # Group consecutive artifact indices
        groups = []
        current_group = [artifact_indices[0]]
        for idx in artifact_indices[1:]:
            if idx == current_group[-1] + 1:
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]
        groups.append(current_group)
        
        # Interpolate each group
        for group in groups:
            start = max(0, group[0] - 1)
            end = min(len(ppg) - 1, group[-1] + 1)
            if start < end and not artifact_mask[start] and not artifact_mask[end]:
                clean_ppg[group] = np.interp(group, [start, end], [ppg[start], ppg[end]])
        
        return clean_ppg
    
    def assess_quality(self, ppg: np.ndarray, filtered_ppg: Optional[np.ndarray] = None) -> SignalQualityMetrics:
        """Comprehensive signal quality assessment."""
        if filtered_ppg is None:
            filtered_ppg = self.bandpass_filter(ppg)
        
        # SNR estimation
        signal_power = np.var(filtered_ppg)
        noise_power = np.var(ppg - filtered_ppg) if len(ppg) == len(filtered_ppg) else np.var(ppg)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10)) if noise_power > 0 else 30
        
        # Perfusion Index (AC/DC ratio)
        dc = np.mean(ppg)
        ac = np.std(filtered_ppg)
        perfusion_index = (ac / (dc + 1e-8)) * 100
        
        # Saturation detection
        saturation_ratio = self.detect_saturation(ppg)
        
        # Motion artifact ratio
        motion_mask = self.detect_motion_artifacts(ppg)
        motion_artifact_ratio = np.mean(motion_mask)
        
        # Overall quality score (0-100)
        quality = 100
        quality -= min(30, saturation_ratio * 100)
        quality -= min(30, motion_artifact_ratio * 100)
        quality -= max(0, 20 - snr_db)  # Penalize low SNR
        quality -= max(0, 10 - perfusion_index)  # Penalize low perfusion
        quality = max(0, min(100, quality))
        
        quality_flags = {
            'saturated': saturation_ratio > 0.05,
            'motion_artifacts': motion_artifact_ratio > 0.15,
            'low_snr': snr_db < 10,
            'low_perfusion': perfusion_index < 0.5,
        }
        
        is_acceptable = all(not flag for flag in quality_flags.values())
        
        return SignalQualityMetrics(
            snr_db=snr_db,
            perfusion_index=perfusion_index,
            saturation_ratio=saturation_ratio,
            motion_artifact_ratio=motion_artifact_ratio,
            overall_quality=quality,
            is_acceptable=is_acceptable,
            quality_flags=quality_flags
        )
    
    def preprocess(self, ppg: np.ndarray, 
                   apply_filter: bool = True,
                   apply_detrend: bool = True,
                   apply_artifact_removal: bool = True,
                   normalize: str = 'minmax') -> Tuple[np.ndarray, SignalQualityMetrics]:
        """
        Complete preprocessing pipeline.
        
        Args:
            ppg: Raw PPG signal
            apply_filter: Apply bandpass filter
            apply_detrend: Apply baseline detrending
            apply_artifact_removal: Detect and interpolate motion artifacts
            normalize: Normalization method ('minmax', 'zscore', 'none')
            
        Returns:
            Tuple of (processed_signal, quality_metrics)
        """
        # Ensure correct dtype
        ppg = np.asarray(ppg, dtype=np.float32)
        
        # Check signal length
        min_samples = int(self.config.min_signal_length * self.config.fs)
        max_samples = int(self.config.max_signal_length * self.config.fs)
        
        if len(ppg) < min_samples:
            raise ValueError(f"Signal too short: {len(ppg)} samples "
                           f"(minimum {min_samples} at {self.config.fs} Hz)")
        if len(ppg) > max_samples:
            ppg = ppg[:max_samples]
        
        # Initial quality assessment on raw signal
        raw_quality = self.assess_quality(ppg)
        
        # Apply preprocessing steps
        processed = ppg.copy()
        
        if apply_detrend:
            processed = self.detrend_signal(processed)
        
        if apply_filter:
            processed = self.bandpass_filter(processed)
        
        if apply_artifact_removal:
            motion_mask = self.detect_motion_artifacts(processed)
            processed = self.interpolate_artifacts(processed, motion_mask)
            # Re-filter after interpolation
            if apply_filter:
                processed = self.bandpass_filter(processed)
        
        # Normalize
        if normalize == 'minmax':
            processed = self.normalize_minmax(processed)
        elif normalize == 'zscore':
            processed = self.normalize_zscore(processed)
        
        # Final quality assessment
        final_quality = self.assess_quality(ppg, processed)
        
        return processed, final_quality
    
    def segment_signal(self, ppg: np.ndarray, 
                       segment_length: float = 10.0,
                       overlap: float = 0.5) -> np.ndarray:
        """
        Segment signal into overlapping windows.
        
        Args:
            ppg: Preprocessed PPG signal
            segment_length: Length of each segment (seconds)
            overlap: Overlap fraction between segments (0-1)
            
        Returns:
            Array of segments [n_segments, segment_samples]
        """
        segment_samples = int(segment_length * self.config.fs)
        step = int(segment_samples * (1 - overlap))
        
        if len(ppg) < segment_samples:
            return ppg.reshape(1, -1)
        
        n_segments = 1 + (len(ppg) - segment_samples) // step
        segments = np.zeros((n_segments, segment_samples), dtype=np.float32)
        
        for i in range(n_segments):
            start = i * step
            end = start + segment_samples
            segments[i] = ppg[start:end]
        
        return segments


def preprocess_ppg_signal(ppg: np.ndarray, fs: float = 125.0,
                          low_freq: float = 0.5, high_freq: float = 16.0,
                          normalize: str = 'minmax') -> Tuple[np.ndarray, SignalQualityMetrics]:
    """
    Convenience function for quick PPG preprocessing.
    
    Args:
        ppg: Raw PPG signal
        fs: Sampling frequency (Hz)
        low_freq: Bandpass low cutoff (Hz)
        high_freq: Bandpass high cutoff (Hz)
        normalize: Normalization method
        
    Returns:
        Tuple of (processed_signal, quality_metrics)
    """
    config = PreprocessingConfig(fs=fs, low_freq=low_freq, high_freq=high_freq)
    preprocessor = PPGPreprocessor(config)
    return preprocessor.preprocess(ppg, normalize=normalize)