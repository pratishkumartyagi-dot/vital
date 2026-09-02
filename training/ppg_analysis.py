"""Reference PPG analysis pipeline using heartpy (Heart Rate) and RRest (Resp Rate).

This script mirrors the algorithms used by the browser-side TypeScript
module at ``src/app/vital-monitoring/ppg-analysis.ts`` and can be used to
validate the live monitor's outputs offline.

Usage:
    python training/ppg_analysis.py path/to/ppg.csv [--fs 30]

The CSV is expected to have a single column of PPG samples (one per line).
It outputs heart rate (BPM), HRV (SDNN/RMSSD/pNN50), rhythm classification
and respiratory rate in breaths/min.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from statistics import mean, pstdev
from typing import Iterable, List, Sequence, Tuple, Optional

import numpy as np
from scipy import signal as sps

try:
    import heartpy as hp
except ImportError as exc:  # pragma: no cover
    print(
        "heartpy is required: pip install heartpy",
        file=sys.stderr,
    )
    raise

try:
    from RRest import RRest as rrest_module
    HAVE_RREST = True
except Exception:  # pragma: no cover - RRest import may fail on some envs
    HAVE_RREST = False


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 95.0) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean of data."""
    if data.size < 2:
        return np.nan, np.nan
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=data.size, replace=True)
        boot_means.append(np.mean(sample))
    alpha = (100 - ci) / 2
    return float(np.percentile(boot_means, alpha)), float(np.percentile(boot_means, 100 - alpha))


def bootstrap_hrv_ci(rr_ms: np.ndarray, n_bootstrap: int = 1000, ci: float = 95.0) -> dict:
    """Compute bootstrap CIs for HRV metrics (SDNN, RMSSD, pNN50)."""
    if rr_ms.size < 2:
        return {
            'sdnn_ci': (np.nan, np.nan),
            'rmssd_ci': (np.nan, np.nan),
            'pnn50_ci': (np.nan, np.nan),
        }
    
    boot_sdnn = []
    boot_rmssd = []
    boot_pnn50 = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(rr_ms, size=rr_ms.size, replace=True)
        sdnn = np.std(sample, ddof=1)
        successive = np.abs(np.diff(sample))
        rmssd = np.sqrt(np.mean(successive ** 2)) if successive.size > 0 else 0.0
        pnn50 = (np.sum(successive > 50.0) / successive.size * 100.0) if successive.size > 0 else 0.0
        boot_sdnn.append(sdnn)
        boot_rmssd.append(rmssd)
        boot_pnn50.append(pnn50)
    
    alpha = (100 - ci) / 2
    return {
        'sdnn_ci': (float(np.percentile(boot_sdnn, alpha)), float(np.percentile(boot_sdnn, 100 - alpha))),
        'rmssd_ci': (float(np.percentile(boot_rmssd, alpha)), float(np.percentile(boot_rmssd, 100 - alpha))),
        'pnn50_ci': (float(np.percentile(boot_pnn50, alpha)), float(np.percentile(boot_pnn50, 100 - alpha))),
    }


def load_ppg_csv(path: str) -> np.ndarray:
    samples: List[float] = []
    with open(path, "r", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            try:
                samples.append(float(row[0]))
            except ValueError:
                continue
    return np.asarray(samples, dtype=float)


def detrend(signal: np.ndarray) -> np.ndarray:
    if signal.size < 3:
        return signal.copy()
    return sps.detrend(signal, type="linear")


def butter_bandpass(sig: np.ndarray, low: float, high: float, fs: float, order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    low_n = max(low / nyq, 1e-5)
    high_n = min(high / nyq, 0.99999)
    b, a = sps.butter(order, [low_n, high_n], btype="band")
    return sps.filtfilt(b, a, sig)


def classify_rhythm(rr_ms: Sequence[float]) -> str:
    if len(rr_ms) < 5:
        return "unknown"
    rr = np.asarray(rr_ms, dtype=float)
    cv = (np.std(rr, ddof=1) / np.mean(rr)) * 100.0 if np.mean(rr) > 0 else 0.0
    if cv < 10:
        return "regular"
    if cv < 20:
        return "regular"
    return "irregular"


def analyze_heart_heartpy(ppg: np.ndarray, fs: float, n_bootstrap: int = 1000) -> dict:
    """Run heartpy for heart rate + HRV (mirrors fmoessle/heart-rate-monitor)."""
    try:
        wd, _ = hp.process(ppg, sample_rate=fs, high_precision=True)
    except Exception as exc:
        return {"error": f"heartpy failed: {exc}"}
    bpm = float(wd.get("bpm", 0.0))
    rr = np.asarray(wd.get("RR_list", []) or [], dtype=float)
    sdnn = float(np.std(rr, ddof=1)) if rr.size > 1 else 0.0
    successive = np.abs(np.diff(rr)) if rr.size > 1 else np.array([])
    rmssd = float(np.sqrt(np.mean(successive ** 2))) if successive.size else 0.0
    pnn50 = float((np.sum(successive > 50.0) / successive.size) * 100.0) if successive.size else 0.0
    
    # Bootstrap confidence intervals
    hrv_ci = bootstrap_hrv_ci(rr, n_bootstrap=n_bootstrap) if rr.size >= 2 else {
        'sdnn_ci': (np.nan, np.nan),
        'rmssd_ci': (np.nan, np.nan),
        'pnn50_ci': (np.nan, np.nan),
    }
    
    return {
        "bpm": bpm,
        "peaks": [int(x) for x in wd.get("peaklist", [])],
        "rr_ms": [float(x) for x in rr.tolist()],
        "sdnn_ms": sdnn,
        "sdnn_ci": hrv_ci['sdnn_ci'],
        "rmssd_ms": rmssd,
        "rmssd_ci": hrv_ci['rmssd_ci'],
        "pnn50": pnn50,
        "pnn50_ci": hrv_ci['pnn50_ci'],
        "rhythm": classify_rhythm(rr),
        "signal_quality": float(wd.get("snr", 0.0) or 0.0),
    }


def analyze_resp_rrest(ppg: np.ndarray, fs: float) -> dict:
    """Run RRest for respiratory rate (mirrors peterhcharlton/RRest)."""
    if not HAVE_RREST:
        return _analyze_resp_fallback(ppg, fs)
    try:
        out = rrest_module.get_respiration(ppg_signal=ppg, fs=fs)
        candidates: List[float] = []
        if isinstance(out, dict):
            for _, val in out.items():
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 1:
                    arr = np.asarray(val, dtype=float)
                    arr = arr[(arr > 0.5) & (arr < 12.0)]
                    if arr.size:
                        candidates.extend((arr * 60.0).tolist())
        if candidates:
            return {
                "resp_rate_bpm": float(np.median(candidates)),
                "method": "RRest",
                "confidence": 0.8,
            }
        return _analyze_resp_fallback(ppg, fs)
    except Exception as exc:
        print(f"RRest failed ({exc}); using fallback", file=sys.stderr)
        return _analyze_resp_fallback(ppg, fs)


def _analyze_resp_fallback(ppg: np.ndarray, fs: float) -> dict:
    """Browser-style fallback: detrend + lowpass + autocorr/FFT fusion."""
    if ppg.size < int(fs * 10):
        return {"resp_rate_bpm": None, "method": "insufficient-data", "confidence": 0.0}
    detrended = detrend(ppg)
    low = butter_bandpass(detrended, 0.1, 0.5, fs, order=2)
    autocorr = np.correlate(low, low, mode="full")
    autocorr = autocorr[autocorr.size // 2:]
    if autocorr[0] != 0:
        autocorr = autocorr / autocorr[0]
    min_lag = int(fs / 0.5)
    max_lag = min(int(fs / 0.1), autocorr.size - 1)
    if min_lag >= max_lag:
        return {"resp_rate_bpm": None, "method": "fallback", "confidence": 0.0}
    search = autocorr[min_lag:max_lag]
    best_lag = int(np.argmax(search)) + min_lag
    autocorr_bpm = (60.0 * fs) / best_lag if best_lag > 0 else 0.0
    freqs, psd = sps.periodogram(low, fs=fs)
    mask = (freqs >= 0.1) & (freqs <= 0.5)
    if not np.any(mask):
        return {"resp_rate_bpm": autocorr_bpm, "method": "autocorr", "confidence": 0.3}
    idx = int(np.argmax(psd[mask]))
    fft_freq = float(freqs[mask][idx])
    fft_bpm = fft_freq * 60.0
    if autocorr_bpm <= 0:
        fused = fft_bpm
        method = "FFT"
    elif abs(fft_bpm - autocorr_bpm) < 8:
        fused = 0.5 * (autocorr_bpm + fft_bpm)
        method = "autocorr+FFT"
    else:
        fused = 0.5 * (autocorr_bpm + fft_bpm)
        method = "autocorr+FFT(fused)"
    confidence = float(min(1.0, max(autocorr[best_lag], psd[mask][idx] / (np.sum(psd[mask]) + 1e-9))))
    if not (5.0 <= fused <= 50.0):
        return {"resp_rate_bpm": None, "method": method, "confidence": 0.0}
    return {"resp_rate_bpm": float(fused), "method": method, "confidence": confidence}


def analyze(ppg_path: str, fs: float = 30.0, n_bootstrap: int = 1000) -> dict:
    ppg = load_ppg_csv(ppg_path)
    heart = analyze_heart_heartpy(ppg, fs, n_bootstrap=n_bootstrap)
    resp = analyze_resp_rrest(ppg, fs)
    return {"heart": heart, "resp": resp, "fs": fs, "samples": int(ppg.size)}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPG HR + Resp Rate analysis")
    parser.add_argument("csv", help="Path to a single-column PPG CSV file")
    parser.add_argument("--fs", type=float, default=30.0, help="Sampling rate in Hz")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap iterations for HRV CIs")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(args.csv, fs=args.fs, n_bootstrap=args.n_bootstrap)
    print(f"PPG samples: {result['samples']} @ {result['fs']} Hz")
    heart = result["heart"]
    if "error" in heart:
        print(f"Heart analysis error: {heart['error']}")
    else:
        print(f"Heart Rate: {heart['bpm']:.1f} BPM  | Rhythm: {heart['rhythm']}")
        print(f"HRV SDNN: {heart['sdnn_ms']:.1f} ms (95% CI: {heart['sdnn_ci'][0]:.1f}-{heart['sdnn_ci'][1]:.1f})")
        print(f"HRV RMSSD: {heart['rmssd_ms']:.1f} ms (95% CI: {heart['rmssd_ci'][0]:.1f}-{heart['rmssd_ci'][1]:.1f})")
        print(f"HRV pNN50: {heart['pnn50']:.1f}% (95% CI: {heart['pnn50_ci'][0]:.1f}-{heart['pnn50_ci'][1]:.1f})")
        print(f"Peaks detected: {len(heart['peaks'])}")
    resp = result["resp"]
    rr = resp.get("resp_rate_bpm")
    print(f"Respiratory Rate: {rr if rr is not None else 'n/a'} /min  (method={resp.get('method')}, conf={resp.get('confidence', 0):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
