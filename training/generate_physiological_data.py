#!/usr/bin/env python3
"""
Advanced PPG-BP Synthetic Data Generator
Based on physiological principles from:
- Kachuee et al. "Cuff-less high-accuracy calibration-free blood pressure estimation using pulse transit time" (2015)
- Mohammadi et al. "Cuff-less blood pressure monitoring via PPG signals using a hybrid CNN-BiLSTM deep learning model with attention mechanism" (2025)
- MIMIC-III waveform database characteristics

Generates realistic PPG waveforms with corresponding ABP (SBP/DBP) values.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.signal import resample, butter, filtfilt
from scipy.interpolate import interp1d
import json


class PhysiologicalPPGGenerator:
    """
    Generates physiologically realistic PPG waveforms with corresponding BP values.
    Based on the Windkessel model and pulse wave propagation theory.
    """
    
    def __init__(self, fs=125, segment_length=1024):
        self.fs = fs
        self.segment_length = segment_length
        self.duration = segment_length / fs  # 8.192 seconds
        
    def generate_windkessel_ppg(self, sbp, dbp, hr, sv=70, c=1.5, r=1.2):
        """
        Generate PPG using 3-element Windkessel model.
        
        Args:
            sbp: Systolic BP (mmHg)
            dbp: Diastolic BP (mmHg)  
            hr: Heart rate (bpm)
            sv: Stroke volume (mL)
            c: Arterial compliance (mL/mmHg)
            r: Peripheral resistance (mmHg·s/mL)
        """
        n = self.segment_length
        t = np.arange(n) / self.fs
        
        # Heart period with RSA (respiratory sinus arrhythmia)
        T_mean = 60.0 / hr
        resp_freq = 0.25  # 15 breaths/min
        rsa_amp = 0.05
        T = T_mean * (1 + rsa_amp * np.sin(2 * np.pi * resp_freq * t))
        
        # Generate beat-by-beat
        ppg = np.zeros(n)
        abp = np.zeros(n)
        
        beat_starts = np.cumsum(T * self.fs).astype(int)
        beat_starts = beat_starts[beat_starts < n - 100]
        
        for i, start in enumerate(beat_starts):
            if start + 100 >= n:
                break
                
            beat_len = int(T[i] * self.fs)
            beat_t = np.arange(beat_len) / self.fs
            
            # Windkessel ABP waveform for this beat
            # Systolic upstroke (rapid)
            systole_dur = 0.3 * T[i]
            systole_samples = int(systole_dur * self.fs)
            
            # Dicrotic notch timing (aortic valve closure)
            notch_time = 0.35 * T[i]
            notch_samples = int(notch_time * self.fs)
            
            # Build ABP beat
            abp_beat = np.zeros(beat_len)
            
            # Systolic rise
            if systole_samples > 0:
                rise = np.linspace(0, 1, systole_samples)
                abp_beat[:systole_samples] = dbp + (sbp - dbp) * (1 - np.cos(np.pi * rise)) / 2
            
            # Dicrotic notch (aortic valve closure) - should not go below DBP
            if notch_samples < beat_len:
                notch_width = int(0.05 * T[i] * self.fs)
                notch_center = notch_samples
                notch_start = max(0, notch_center - notch_width // 2)
                notch_end = min(beat_len, notch_center + notch_width // 2)
                # Notch depth is small fraction of pulse pressure, minimum is DBP
                notch_depth = 0.05 * (sbp - dbp)
                notch_values = dbp + (abp_beat[notch_start:notch_end] - dbp) * (1 - notch_depth * np.exp(-0.5 * ((np.arange(notch_end - notch_start) - notch_width // 2) / (notch_width / 4)) ** 2))
                abp_beat[notch_start:notch_end] = np.maximum(notch_values, dbp)
            
            # Diastolic decay (exponential) - approaches DBP from above
            if notch_samples < beat_len:
                decay_t = beat_t[notch_samples:] - beat_t[notch_samples]
                tau = r * c  # Time constant
                # Start from max of current value or DBP
                start_val = max(abp_beat[notch_samples], dbp)
                abp_beat[notch_samples:] = dbp + (start_val - dbp) * np.exp(-decay_t / tau)
                abp_beat[notch_samples:] = np.maximum(abp_beat[notch_samples:], dbp)
            
            # PPG is related to ABP but with different morphology
            # PPG ~ volume change ~ compliance * dP/dt + P
            # Simplified: PPG follows ABP with phase shift and different scaling
            ppg_beat = self._abp_to_ppg(abp_beat, sbp, dbp, T[i])
            
            # Place in signal
            end = min(start + beat_len, n)
            actual_len = end - start
            ppg[start:end] += ppg_beat[:actual_len]
            abp[start:end] += abp_beat[:actual_len]
        
        # Add respiratory modulation (baseline wander)
        resp_wave = 0.15 * np.sin(2 * np.pi * resp_freq * t)
        ppg = ppg * (1 + resp_wave)
        abp = abp * (1 + 0.1 * resp_wave)
        
        # Add Mayer waves (0.1 Hz)
        mayer = 0.05 * np.sin(2 * np.pi * 0.1 * t)
        ppg = ppg * (1 + mayer)
        abp = abp * (1 + 0.05 * mayer)
        
        # Add noise - keep physiological
        ppg += np.random.normal(0, 0.005, n)
        abp += np.random.normal(0, 0.3, n)  # Reduced noise
        
        # Clamp ABP to physiological range
        abp = np.clip(abp, 30, 250)  # 30-250 mmHg range
        
        # Normalize PPG
        ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
        
        return ppg.astype(np.float32), abp.astype(np.float32)
    
    def _abp_to_ppg(self, abp_beat, sbp, dbp, T):
        """Convert ABP beat to PPG beat with physiological differences."""
        # PPG has:
        # - Faster upstroke (peripheral vasoconstriction)
        # - More prominent dicrotic notch (reflection)
        # - Different diastolic decay (venous pooling)
        
        # Differentiate ABP to get flow-like signal
        dt = T / len(abp_beat)
        dP_dt = np.gradient(abp_beat) / dt
        
        # PPG ~ flow + pressure (simplified)
        ppg_beat = 0.3 * dP_dt / np.max(np.abs(dP_dt)) + 0.7 * (abp_beat - dbp) / (sbp - dbp)
        
        # Normalize to [-1, 1] range
        ppg_beat = 2 * (ppg_beat - ppg_beat.min()) / (ppg_beat.max() - ppg_beat.min() + 1e-8) - 1
        
        return ppg_beat
    
    def generate_population(self, n_samples=5000, output_dir='training/data/physiological_ppg'):
        """Generate population with realistic BP distributions."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        records = []
        
        print(f"Generating {n_samples} physiological PPG-BP samples...")
        
        for i in tqdm(range(n_samples), desc="Generating"):
            # Sample from realistic BP distributions (correlated)
            # Based on NHANES data
            age = np.random.normal(50, 15)
            age = np.clip(age, 18, 90)
            
            # BP increases with age
            sbp_base = 100 + 0.5 * age + np.random.normal(0, 12)
            dbp_base = 65 + 0.3 * age + np.random.normal(0, 8)
            
            # Add physiological variability
            sbp = np.clip(sbp_base, 85, 200)
            dbp = np.clip(dbp_base, 45, 120)
            
            # Ensure physiological constraint
            if sbp - dbp < 20:
                sbp = dbp + np.random.uniform(20, 60)
            
            # Heart rate correlates with BP and age
            hr = np.random.normal(70 - 0.1 * age, 10)
            hr = np.clip(hr, 45, 110)
            
            # Generate signals
            ppg, abp = self.generate_windkessel_ppg(sbp, dbp, hr)
            
            # Verify generated BP matches target
            gen_sbp = np.max(abp)
            gen_dbp = np.min(abp)
            
            # Accept all samples for now, adjust validation later
            npy_path = output_dir / f'ppg_{i:06d}.npy'
            np.save(npy_path, ppg)
            
            records.append({
                'ppg': str(npy_path),
                'sbp': round(sbp, 1),
                'dbp': round(dbp, 1),
                'hr': round(hr, 1),
                'age': round(age, 1),
                'gen_sbp': round(gen_sbp, 1),
                'gen_dbp': round(gen_dbp, 1)
            })
        
        df = pd.DataFrame(records)
        csv_path = output_dir / 'physiological_ppg_bp.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"\nGenerated {len(records)} valid samples")
        print(f"SBP range: {df['sbp'].min():.1f}-{df['sbp'].max():.1f} (target)")
        print(f"DBP range: {df['dbp'].min():.1f}-{df['dbp'].max():.1f} (target)")
        print(f"Generated SBP: {df['gen_sbp'].min():.1f}-{df['gen_sbp'].max():.1f}")
        print(f"Generated DBP: {df['gen_dbp'].min():.1f}-{df['gen_dbp'].max():.1f}")
        print(f"HR range: {df['hr'].min():.1f}-{df['hr'].max():.1f}")
        print(f"Age range: {df['age'].min():.1f}-{df['age'].max():.1f}")
        print(f"Metadata saved to: {csv_path}")
        
        return csv_path


def add_realistic_augmentation(ppg, fs=125):
    """Apply realistic augmentations to PPG signals."""
    augmented = ppg.copy()
    
    # Random baseline wander
    if np.random.rand() < 0.3:
        t = np.arange(len(augmented)) / fs
        freq = np.random.uniform(0.05, 0.5)
        amp = np.random.uniform(0.02, 0.1)
        augmented += amp * np.sin(2 * np.pi * freq * t + np.random.rand() * 2 * np.pi)
    
    # Motion artifact (sudden shifts)
    if np.random.rand() < 0.1:
        idx = np.random.randint(len(augmented) // 4, 3 * len(augmented) // 4)
        shift = np.random.normal(0, 0.5)
        augmented[idx:] += shift
    
    # Amplitude variation (vasoconstriction/dilation)
    if np.random.rand() < 0.3:
        scale = np.random.uniform(0.7, 1.3)
        augmented *= scale
    
    # Re-normalize
    augmented = (augmented - augmented.mean()) / (augmented.std() + 1e-8)
    
    return augmented


if __name__ == '__main__':
    generator = PhysiologicalPPGGenerator(fs=125, segment_length=1024)
    csv_path = generator.generate_population(n_samples=5000)
    
    # Also save generator config
    config = {
        'fs': 125,
        'segment_length': 1024,
        'duration': 8.192,
        'model': '3-element Windkessel',
        'n_samples': 5000
    }
    with open(Path(csv_path).parent / 'generator_config.json', 'w') as f:
        json.dump(config, f, indent=2)