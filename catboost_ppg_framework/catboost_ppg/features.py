"""
PPG Feature Extraction Module
============================

Extracts comprehensive features from preprocessed PPG signals for CatBoost modeling:
- Waveform Morphology features (peaks, valleys, widths, areas, derivatives)
- Frequency Domain features (FFT, PSD, bandpowers)
- HRV (Heart Rate Variability) features from Inter-Beat Intervals
- Time-domain statistical features
"""

import numpy as np
import scipy.signal as signal
from scipy.signal import find_peaks, welch
from scipy.stats import skew, kurtosis
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


@dataclass
class MorphologyFeatures:
    """Waveform morphology features."""
    # Peak-related
    num_peaks: float
    mean_peak_amplitude: float
    std_peak_amplitude: float
    peak_amplitude_ratio: float
    
    # Pulse characteristics
    pulse_width: float  # at 50% height
    rise_time: float
    fall_time: float
    
    # Areas and ratios
    systolic_area: float
    diastolic_area: float
    area_ratio: float
    p2p_amplitude: float
    
    # Derivative features (first derivative dPPG)
    max_dppg: float
    min_dppg: float
    dppg_peak_ratio: float
    
    # Augmentation-related (if available)
    augmentation_index: float
    dicrotic_notch_present: float
    
    # Statistical
    skewness: float
    kurtosis_val: float


@dataclass
class FrequencyFeatures:
    """Frequency domain features."""
    # FFT-based
    dominant_freq: float  # Hz
    dominant_power: float
    total_power: float
    
    # Bandpowers (cardiac + respiratory bands)
    hf_power: float  # 0.15-0.4 Hz (HRV HF)
    lf_power: float  # 0.04-0.15 Hz (HRV LF)
    vlf_power: float  # 0.003-0.04 Hz (HRV VLF)
    lf_hf_ratio: float
    
    # PPG-specific frequency features
    mean_freq: float
    median_freq: float
    spectral_centroid: float
    spectral_entropy: float


@dataclass
class HRVFeatures:
    """Heart Rate Variability features from IBI."""
    # Time-domain
    mean_ibi: float
    sdnn: float  # Standard deviation of NN intervals
    rmssd: float  # Root mean square of successive differences
    pnn50: float  # % of successive differences > 50ms
    pnn20: float
    hr_mean: float
    hr_std: float
    hr_min: float
    hr_max: float
    
    # Statistical
    ibi_skewness: float
    ibi_kurtosis: float
    
    # Non-linear (simple)
    sd1: float
    sd2: float
    sd_ratio: float


@dataclass
class PPGFeatures:
    """Complete set of PPG features."""
    morphology: MorphologyFeatures
    frequency: FrequencyFeatures
    hrv: HRVFeatures
    
    # Additional computed features
    signal_duration: float
    sample_count: float
    fs: float
    
    @property
    def all_features(self) -> Dict[str, float]:
        """Flatten all features into a dictionary."""
        result = {}
        
        # Morphology
        for k, v in asdict(self.morphology).items():
            result[f"morph_{k}"] = float(v)
        
        # Frequency
        for k, v in asdict(self.frequency).items():
            result[f"freq_{k}"] = float(v)
        
        # HRV
        for k, v in asdict(self.hrv).items():
            result[f"hrv_{k}"] = float(v)
        
        # Meta
        result["signal_duration"] = float(self.signal_duration)
        result["sample_count"] = float(self.sample_count)
        
        return result


class PPGFeatureExtractor:
    """
    Extracts comprehensive PPG features.
    """
    
    def __init__(self, fs: float = 125.0):
        self.fs = fs
        self.min_peaks_distance = int(fs * 0.3)  # Min distance for valid peaks (>= 0.3s, < 200 BPM)
        self.max_peaks_distance = int(fs * 1.5)  # Max distance for valid peaks (> 40 BPM)
    
    def extract_peaks_and_valleys(self, ppg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract peak and valley indices from PPG signal.
        """
        if len(ppg) < 10:
            return np.array([]), np.array([])
        
        # Find peaks (systolic peaks)
        peaks, peak_props = find_peaks(
            ppg,
            distance=self.min_peaks_distance,
            height=np.mean(ppg) + 0.3 * np.std(ppg),
            prominence=0.1 * np.max(ppg)
        )
        
        # Find valleys
        valleys, valley_props = find_peaks(
            -ppg,
            distance=self.min_peaks_distance
        )
        
        return peaks, valleys
    
    def extract_interbeat_intervals(self, peaks: np.ndarray) -> np.ndarray:
        """Extract Inter-Beat Intervals (IBI) in milliseconds."""
        if len(peaks) < 2:
            return np.array([])
        ibis = (peaks[1:] - peaks[:-1]) * 1000.0 / self.fs
        # Filter physiologically plausible IBIs (300-2000ms)
        ibis = ibis[(ibis > 300) & (ibis < 2000)]
        return ibis
    
    def extract_morphology_features(self, ppg: np.ndarray, 
                                   peaks: np.ndarray, 
                                   valleys: np.ndarray) -> MorphologyFeatures:
        """Extract waveform morphology features."""
        if len(peaks) == 0 or len(ppg) == 0:
            return MorphologyFeatures(
                num_peaks=0, mean_peak_amplitude=0, std_peak_amplitude=0, peak_amplitude_ratio=0,
                pulse_width=0, rise_time=0, fall_time=0, systolic_area=0, diastolic_area=0,
                area_ratio=0, p2p_amplitude=0, max_dppg=0, min_dppg=0, dppg_peak_ratio=0,
                augmentation_index=0, dicrotic_notch_present=0, skewness=0, kurtosis_val=0
            )
        
        peak_heights = ppg[peaks]
        
        # Compute 1st derivative (dPPG)
        dppg = np.gradient(ppg)
        dppg_peaks, _ = find_peaks(dppg, distance=self.min_peaks_distance)
        dppg_valleys, _ = find_peaks(-dppg, distance=self.min_peaks_distance)
        
        # Peak amplitudes
        num_peaks = float(len(peaks))
        mean_peak_amplitude = float(np.mean(peak_heights))
        std_peak_amplitude = float(np.std(peak_heights))
        p2p_amplitude = float(np.max(ppg) - np.min(ppg))
        peak_amplitude_ratio = float(mean_peak_amplitude / (p2p_amplitude + 1e-8))
        
        # Average characteristics across beats
        pulse_widths = []
        rise_times = []
        fall_times = []
        systolic_areas = []
        diastolic_areas = []
        
        for i, peak_idx in enumerate(peaks[:min(5, len(peaks))]):  # Use first few beats
            # Find corresponding valley before and after
            if i == 0:
                prev_valley = valleys[valleys < peak_idx]
                next_valley = valleys[valleys > peak_idx]
            else:
                # Simple approach: find nearest valleys
                pass
            
            # Estimate pulse width at 50% height
            half_height = ppg[peak_idx] - 0.5 * p2p_amplitude
            if half_height < 0:
                half_height = np.mean(ppg[peaks])
            
            # Rise time (valley to peak)
            # Fall time (peak to next valley)
            if len(valleys) > 0:
                nearby_valleys = valleys[np.abs(valleys - peak_idx) < self.fs * 1.0]
                if len(nearby_valleys) >= 2:
                    left_v = nearby_valleys[nearby_valleys < peak_idx].max()
                    right_v = nearby_valleys[nearby_valleys > peak_idx].min()
                    rise_times.append((peak_idx - left_v) / self.fs)
                    fall_times.append((right_v - peak_idx) / self.fs)
                    pulse_widths.append((right_v - left_v) / self.fs)
                    
                    # Area under curve
                    systolic_areas.append(np.trapz(ppg[left_v:peak_idx+1], dx=1/self.fs))
                    diastolic_areas.append(np.trapz(ppg[peak_idx:right_v+1], dx=1/self.fs))
        
        pulse_width = float(np.mean(pulse_widths)) if pulse_widths else 0.0
        rise_time = float(np.mean(rise_times)) if rise_times else 0.0
        fall_time = float(np.mean(fall_times)) if fall_times else 0.0
        systolic_area = float(np.mean(systolic_areas)) if systolic_areas else 0.0
        diastolic_area = float(np.mean(diastolic_areas)) if diastolic_areas else 0.0
        area_ratio = float(systolic_area / (diastolic_area + 1e-8)) if diastolic_area > 0 else 0.0
        
        # Derivative features
        max_dppg = float(np.max(dppg)) if len(dppg) > 0 else 0.0
        min_dppg = float(np.min(dppg)) if len(dppg) > 0 else 0.0
        dppg_peak_ratio = float(max_dppg / (-min_dppg + 1e-8)) if min_dppg < 0 else 0.0
        
        # Statistical features
        skewness_val = float(skew(ppg))
        kurtosis_val = float(kurtosis(ppg))
        
        # Augmentation and dicrotic notch (simplified heuristics)
        augmentation_index = 0.0
        dicrotic_notch_present = float(1.0) if len(dppg_valleys) > len(peaks) else float(0.0)
        
        return MorphologyFeatures(
            num_peaks=num_peaks,
            mean_peak_amplitude=mean_peak_amplitude,
            std_peak_amplitude=std_peak_amplitude,
            peak_amplitude_ratio=peak_amplitude_ratio,
            pulse_width=pulse_width,
            rise_time=rise_time,
            fall_time=fall_time,
            systolic_area=systolic_area,
            diastolic_area=diastolic_area,
            area_ratio=area_ratio,
            p2p_amplitude=p2p_amplitude,
            max_dppg=max_dppg,
            min_dppg=min_dppg,
            dppg_peak_ratio=dppg_peak_ratio,
            augmentation_index=augmentation_index,
            dicrotic_notch_present=dicrotic_notch_present,
            skewness=skewness_val,
            kurtosis_val=kurtosis_val
        )
    
    def extract_frequency_features(self, ppg: np.ndarray) -> FrequencyFeatures:
        """Extract frequency domain features."""
        if len(ppg) < 16:
            return FrequencyFeatures(
                dominant_freq=0, dominant_power=0, total_power=0,
                hf_power=0, lf_power=0, vlf_power=0, lf_hf_ratio=0,
                mean_freq=0, median_freq=0, spectral_centroid=0, spectral_entropy=0
            )
        
        nperseg = min(len(ppg), 256)
        try:
            freqs, psd = welch(ppg, fs=self.fs, nperseg=nperseg, scaling='density')
        except Exception:
            # Fallback to FFT
            n = len(ppg)
            fft = np.abs(np.fft.rfft(ppg)) ** 2
            freqs = np.fft.rfftfreq(n, 1/self.fs)
            psd = fft / n
        
        # Compute total power
        df = freqs[1] - freqs[0]
        total_power = float(np.sum(psd) * df)
        
        # Find dominant frequency (cardiac band 0.5-3.5 Hz)
        cardiac_mask = (freqs >= 0.5) & (freqs <= 3.5)
        if np.any(cardiac_mask):
            idx = np.argmax(psd[cardiac_mask])
            freq_indices = np.where(cardiac_mask)[0]
            dominant_freq = float(freqs[freq_indices[idx]])
            dominant_power = float(psd[freq_indices[idx]])
        else:
            idx = np.argmax(psd)
            dominant_freq = float(freqs[idx])
            dominant_power = float(psd[idx])
        
        # HRV frequency bands
        vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
        
        vlf_power = float(np.sum(psd[vlf_mask]) * df) if np.any(vlf_mask) else 0.0
        lf_power = float(np.sum(psd[lf_mask]) * df) if np.any(lf_mask) else 0.0
        hf_power = float(np.sum(psd[hf_mask]) * df) if np.any(hf_mask) else 0.0
        
        lf_hf_ratio = float(lf_power / (hf_power + 1e-8))
        
        # Spectral moments
        mean_freq = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-8))
        
        # Median frequency
        cumsum = np.cumsum(psd)
        cumsum_norm = cumsum / (cumsum[-1] + 1e-8)
        median_freq = float(freqs[np.where(cumsum_norm >= 0.5)[0][0]]) if len(freqs) > 0 else 0.0
        
        # Spectral centroid
        spectral_centroid = mean_freq
        
        # Spectral entropy
        psd_norm = psd / (np.sum(psd) + 1e-8)
        spectral_entropy = float(-np.sum(psd_norm * np.log(psd_norm + 1e-10)) / 
                               (np.log(len(psd)) + 1e-10)) if len(psd) > 1 else 0.0
        
        return FrequencyFeatures(
            dominant_freq=dominant_freq,
            dominant_power=dominant_power,
            total_power=total_power,
            hf_power=hf_power,
            lf_power=lf_power,
            vlf_power=vlf_power,
            lf_hf_ratio=lf_hf_ratio,
            mean_freq=mean_freq,
            median_freq=median_freq,
            spectral_centroid=spectral_centroid,
            spectral_entropy=spectral_entropy
        )
    
    def extract_hrv_features(self, ppg: np.ndarray, peaks: np.ndarray) -> HRVFeatures:
        """Extract HRV features from IBI."""
        ibis = self.extract_interbeat_intervals(peaks)
        
        if len(ibis) < 2:
            return HRVFeatures(
                mean_ibi=0, sdnn=0, rmssd=0, pnn50=0, pnn20=0,
                hr_mean=0, hr_std=0, hr_min=0, hr_max=0,
                ibi_skewness=0, ibi_kurtosis=0, sd1=0, sd2=0, sd_ratio=0
            )
        
        # Time-domain HRV
        mean_ibi = float(np.mean(ibis))
        sdnn = float(np.std(ibis, ddof=1))  # Standard deviation of NN intervals
        
        # Successive differences
        diff_ibis = np.diff(ibis)
        rmssd = float(np.sqrt(np.mean(diff_ibis ** 2)))
        
        pnn50 = float(100.0 * np.sum(np.abs(diff_ibis) > 50) / len(diff_ibis))
        pnn20 = float(100.0 * np.sum(np.abs(diff_ibis) > 20) / len(diff_ibis))
        
        # Heart rate
        hr = 60000.0 / ibis
        hr_mean = float(np.mean(hr))
        hr_std = float(np.std(hr, ddof=1))
        hr_min = float(np.min(hr))
        hr_max = float(np.max(hr))
        
        # Statistical
        ibi_skewness = float(skew(ibis))
        ibi_kurtosis = float(kurtosis(ibis))
        
        # Poincaré plot features
        if len(diff_ibis) > 0:
            sd1 = float(np.sqrt(0.5 * np.var(diff_ibis)))
            sd2 = float(np.sqrt(2 * np.var(ibis[:-1]) - 0.5 * np.var(diff_ibis)))
        else:
            sd1, sd2 = 0.0, 0.0
        sd_ratio = float(sd2 / (sd1 + 1e-8))
        
        return HRVFeatures(
            mean_ibi=mean_ibi,
            sdnn=sdnn,
            rmssd=rmssd,
            pnn50=pnn50,
            pnn20=pnn20,
            hr_mean=hr_mean,
            hr_std=hr_std,
            hr_min=hr_min,
            hr_max=hr_max,
            ibi_skewness=ibi_skewness,
            ibi_kurtosis=ibi_kurtosis,
            sd1=sd1,
            sd2=sd2,
            sd_ratio=sd_ratio
        )
    
    def extract_features(self, ppg: np.ndarray) -> PPGFeatures:
        """
        Extract all features from PPG signal.
        
        Args:
            ppg: Preprocessed PPG signal (1D numpy array)
            
        Returns:
            PPGFeatures object
        """
        # Ensure correct format
        ppg = np.asarray(ppg, dtype=np.float64)
        
        if len(ppg) == 0:
            raise ValueError("Empty PPG signal")
        
        # Extract peaks and valleys
        peaks, valleys = self.extract_peaks_and_valleys(ppg)
        
        # Extract all feature groups
        morphology = self.extract_morphology_features(ppg, peaks, valleys)
        frequency = self.extract_frequency_features(ppg)
        hrv = self.extract_hrv_features(ppg, peaks)
        
        return PPGFeatures(
            morphology=morphology,
            frequency=frequency,
            hrv=hrv,
            signal_duration=float(len(ppg) / self.fs),
            sample_count=float(len(ppg)),
            fs=float(self.fs)
        )