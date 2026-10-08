#!/usr/bin/env python3
"""
Generate synthetic PPG-BP data for BPNet training demo.
Creates realistic PPG waveforms with corresponding SBP/DBP values.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def generate_realistic_ppg(fs=125, duration=8.192, heart_rate=70, systolic=120, diastolic=80):
    """
    Generate a realistic PPG waveform with given BP targets.
    Based on physiological relationships between PPG morphology and BP.
    """
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    
    # Heart period
    T = 60.0 / heart_rate
    
    # Base PPG shape: systolic upstroke + dicrotic notch + diastolic decay
    def ppg_beat(t_local):
        # Normalized beat period [0, 1]
        phase = (t_local % T) / T
        
        # Systolic upstroke (rapid rise)
        systolic_phase = phase < 0.35
        systolic = np.where(systolic_phase, 
                           np.sin(np.pi * phase / 0.35) ** 1.5, 0)
        
        # Dicrotic notch (reflection wave)
        notch_phase = (phase >= 0.35) & (phase < 0.55)
        notch = np.where(notch_phase,
                        0.3 * np.exp(-((phase - 0.35) / 0.05) ** 2), 0)
        
        # Diastolic decay
        diastolic_phase = phase >= 0.55
        diastolic = np.where(diastolic_phase,
                            0.15 * np.exp(-(phase - 0.55) / 0.2), 0)
        
        return systolic + notch + diastolic
    
    # Generate beat-by-beat with variability
    ppg = np.zeros(n_samples)
    for i in range(n_samples):
        # Add respiratory sinus arrhythmia (RSA)
        rsa = 0.02 * np.sin(2 * np.pi * 0.25 * t[i])  # 0.25 Hz respiration
        local_hr = heart_rate * (1 + rsa)
        local_T = 60.0 / local_hr
        ppg[i] = ppg_beat(t[i]) * (1 + 0.05 * np.sin(2 * np.pi * 0.1 * t[i]))
    
    # Scale amplitude based on pulse pressure (SBP - DBP)
    pulse_pressure = systolic - diastolic
    amplitude_scale = 0.5 + pulse_pressure / 150.0  # Larger PP -> larger AC
    ppg = ppg * amplitude_scale
    
    # Add baseline wander (respiratory + Mayer waves)
    baseline = (0.1 * np.sin(2 * np.pi * 0.25 * t) +  # Respiratory
                0.05 * np.sin(2 * np.pi * 0.1 * t))   # Mayer wave
    ppg = ppg + baseline
    
    # Add noise
    ppg = ppg + np.random.normal(0, 0.01, n_samples)
    
    # Normalize
    ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
    
    return ppg.astype(np.float32)


def generate_dataset(num_samples=2000, output_dir='training/data/synthetic_ppg'):
    """Generate synthetic PPG-BP dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    
    print(f"Generating {num_samples} synthetic PPG-BP samples...")
    
    for i in tqdm(range(num_samples), desc="Generating"):
        # Sample realistic BP values (correlated)
        # SBP ~ N(120, 15), DBP ~ N(80, 10) with correlation
        sbp = np.random.normal(120, 15)
        dbp = np.random.normal(80, 10)
        
        # Ensure physiological constraints
        sbp = np.clip(sbp, 90, 180)
        dbp = np.clip(dbp, 50, 110)
        if sbp <= dbp:
            sbp = dbp + np.random.uniform(15, 40)
        
        # Heart rate correlates with BP
        hr = np.random.normal(70, 10)
        hr = np.clip(hr, 50, 100)
        
        # Generate PPG
        ppg = generate_realistic_ppg(
            fs=125, duration=8.192,  # 1024 samples
            heart_rate=hr, systolic=sbp, diastolic=dbp
        )
        
        # Save as .npy
        npy_path = output_dir / f'ppg_{i:05d}.npy'
        np.save(npy_path, ppg)
        
        records.append({
            'ppg': str(npy_path),
            'sbp': round(sbp, 1),
            'dbp': round(dbp, 1),
            'hr': round(hr, 1)
        })
    
    # Save CSV metadata
    df = pd.DataFrame(records)
    csv_path = output_dir / 'synthetic_ppg_bp.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"\nGenerated {len(records)} samples")
    print(f"SBP range: {df['sbp'].min():.1f}-{df['sbp'].max():.1f}")
    print(f"DBP range: {df['dbp'].min():.1f}-{df['dbp'].max():.1f}")
    print(f"HR range: {df['hr'].min():.1f}-{df['hr'].max():.1f}")
    print(f"Metadata saved to: {csv_path}")
    
    return csv_path


if __name__ == '__main__':
    generate_dataset(num_samples=2000)