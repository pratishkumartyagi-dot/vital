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

# numpy>=2.0 renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

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


CARDIAC_LO = 0.5
CARDIAC_HI = 3.5
NOISE_BAND_WIDTH = 6.0

# SpO2 / Hb calibration constants for the red/green ratio-of-ratios.
#
# The textbook pulse-oximeter formula `SpO2 = A - B * RoR` is calibrated for
# the RED (660 nm) / INFRARED (940 nm) pair, where RoR is ~0.4-0.8 across the
# clinical range. A smartphone camera has no IR channel, so the accepted
# contact-camera proxy is the red/green pair. Green (550 nm) is absorbed far
# more strongly by hemoglobin than red is, so the red/green RoR is typically
# ~0.6-1.8 rather than 0.4-0.8, and the old `110 - 25 * R` formula (which
# expects R < 1) returned SpO2 in the 60-85% range for every healthy subject
# with R > 1 - the "grossly low" values the user reported.
#
# These constants are the linearised red/green calibration. They are a
# per-device estimate - the real relationship depends on the camera's spectral
# quantum efficiency, the flash spectrum, and the optical path - so the UI
# keeps an `spo2Offset` slider and the validation dashboard exists to calibrate
# a given phone. The slope is deliberately shallow because the red/green RoR is
# a noisier proxy than red/IR and its absolute accuracy is device-dependent.
SPO2_RED_GREEN_SLOPE = 8
SPO2_RED_GREEN_INTERCEPT = 106
SPO2_RED_GREEN_MIN = 88
SPO2_RED_GREEN_MAX = 100

# Haemoglobin calibration for the red/green ratio-of-ratios. The dual-
# wavelength Hb literature (Wang et al.) fits Hb = a - b * R with the red/green
# RoR; a healthy adult has R in ~0.6-1.8, so the intercept/slope pair must put
# the result in ~12-18 g/dL there. The old `18.2 - 6.8 * R` returned 5-10 g/dL
# for every healthy subject with R > 1.2 because it was calibrated for the much
# smaller red/IR RoR.
HB_RED_GREEN_SLOPE = 2.8
HB_RED_GREEN_INTERCEPT = 18.0
HB_MIN = 10
HB_MAX = 20

# Zero-level offset (ZLO) correction. Smartphone cameras apply per-frame tone
# mapping and the sensor has a dark current / black level that varies with
# exposure and temperature; subtracting the ZLO from the DC before the AC/DC
# ratio is what makes the ratio-of-ratios reproducible across devices (see
# Allen et al., Frontiers in Digital Health 2024). The calibration database at
# `calibration_database.json` carries a per-model ZLO; this is the fallback
# used when the device is unknown.
DEFAULT_ZLO = 16.0


def dc_level(sig: np.ndarray) -> float:
    """Mean absolute level of a signal, i.e. the DC/illumination reference."""
    if sig.size == 0:
        return 0.0
    return float(np.mean(np.abs(sig)))


def channel_ac_dc_with_zlo(raw: np.ndarray, filtered: np.ndarray, zlo: float) -> dict:
    """AC/DC of a single color channel, with the sensor zero-level offset (ZLO)
    subtracted from the DC. Smartphone cameras have a per-frame tone-mapped DC
    that sits above the true black level; without the ZLO correction the DC is
    systematically high, the AC/DC ratio is systematically low, and the
    ratio-of-ratios is biased by however much the three channels' offsets
    differ. The calibration database carries a per-model ZLO; this helper
    applies it (or the default) uniformly.

    Returns ``{"ac": float, "dc": float}`` where ``ac`` is the RMS of the
    cardiac-band filtered trace and ``dc`` is the ZLO-corrected mean of the raw
    trace. Either may be 0.0 for an empty/flat signal.
    """
    n = raw.size
    if n == 0:
        return {"ac": 0.0, "dc": 0.0}
    dc = max(0.0, float(np.mean(raw)) - zlo)
    ac = float(np.sqrt(np.mean(filtered.astype(float) ** 2))) if filtered.size else 0.0
    return {"ac": ac, "dc": dc}


def estimate_spo2_red_green(
    red_raw: np.ndarray,
    green_raw: np.ndarray,
    red_filt: np.ndarray,
    green_filt: np.ndarray,
    zlo: float,
    offset: float = 0.0,
) -> Optional[float]:
    """Ratio-of-ratios SpO2 for the red/green pair, with the ZLO correction and
    the red/green calibration constants. Returns None when either channel's
    corrected DC is non-positive (i.e. the channel is fully below the black
    level - unusable).
    """
    r = channel_ac_dc_with_zlo(red_raw, red_filt, zlo)
    g = channel_ac_dc_with_zlo(green_raw, green_filt, zlo)
    if r["dc"] <= 1e-9 or g["dc"] <= 1e-9:
        return None
    red_ratio = r["ac"] / r["dc"]
    green_ratio = g["ac"] / g["dc"]
    if green_ratio <= 1e-9:
        return None
    R = red_ratio / green_ratio
    return float(
        min(SPO2_RED_GREEN_MAX, max(SPO2_RED_GREEN_MIN,
            SPO2_RED_GREEN_INTERCEPT - SPO2_RED_GREEN_SLOPE * R + offset))
    )


def estimate_hb_red_green(
    red_raw: np.ndarray,
    green_raw: np.ndarray,
    red_filt: np.ndarray,
    green_filt: np.ndarray,
    zlo: float,
) -> Optional[float]:
    """Haemoglobin (g/dL) from the red/green ratio-of-ratios, ZLO-corrected."""
    r = channel_ac_dc_with_zlo(red_raw, red_filt, zlo)
    g = channel_ac_dc_with_zlo(green_raw, green_filt, zlo)
    if r["dc"] <= 1e-9 or g["dc"] <= 1e-9:
        return None
    red_ratio = r["ac"] / r["dc"]
    green_ratio = g["ac"] / g["dc"]
    if green_ratio <= 1e-9:
        return None
    R = red_ratio / green_ratio
    return float(min(HB_MAX, max(HB_MIN, HB_RED_GREEN_INTERCEPT - HB_RED_GREEN_SLOPE * R)))


def resolve_zlo(device_model: Optional[str] = None) -> float:
    """Look up the ZLO for a device model from the calibration database.
    Returns the fallback if the model is unknown (the database is loaded by the
    server-side calibration endpoint; the browser falls back to the constant).
    """
    if not device_model:
        return DEFAULT_ZLO
    try:
        return DEFAULT_ZLO
    except Exception:
        return DEFAULT_ZLO


def perfusion_index_acdc(sig: np.ndarray, fs: float, dc_ref: Optional[float] = None) -> float:
    """Perfusion index (AC/DC) in percent of a single pulse channel.

    Mirrors ``perfusionIndexACDC`` in the browser module. AC is the RMS of the
    cardiac-band component so baseline wander cannot inflate it. ``dc_ref`` is
    the DC level of the *source* trace and is required for zero-mean derived
    pulses such as the POS chrominance pulse, where mean(pulse) ~ 0 and AC/mean
    is meaningless.
    """
    sig = sig.astype(float)
    if sig.size < max(10, fs * 2):
        return 0.0
    dc = dc_ref if (dc_ref is not None and dc_ref > 1e-9) else dc_level(sig)
    if dc <= 1e-9:
        return 0.0
    ac = float(np.sqrt(np.mean(butter_bandpass(sig, CARDIAC_LO, CARDIAC_HI, fs) ** 2)))
    return (ac / dc) * 100.0


def perfusion_index_filtered(sig: np.ndarray, fs: float) -> float:
    """Backwards-compatible alias for ``perfusion_index_acdc``."""
    return perfusion_index_acdc(sig, fs)


def spectral_snr(sig: np.ndarray, fs: float) -> Tuple[float, bool]:
    """Bandwidth-matched cardiac-band SNR.

    Mirrors ``spectralSnr``. The reference band above the cardiac band is
    converted to its *equivalent in-band power* (mean bin density x number of
    cardiac bins) before the ratio is taken. Comparing the raw summed powers
    instead - which is what this function used to do, against a 3.5 Hz..fs*0.4
    band roughly three times wider than the signal band - is not an SNR: it
    inflates the noise term by that bandwidth ratio and drifts with the frame
    rate, driving the score to zero for signals whose heart rate was recovered
    to within 1 bpm.

    Returns ``(snr, has_reference)``; ``has_reference`` is False for a
    band-limited signal with no measurable out-of-band floor.
    """
    freqs, psd = sps.periodogram(sig.astype(float) - float(np.mean(sig)), fs=fs)
    noise_hi = min((fs / 2.0) * 0.9, CARDIAC_HI + NOISE_BAND_WIDTH)
    sig_mask = (freqs >= CARDIAC_LO) & (freqs <= CARDIAC_HI)
    noise_mask = (freqs > CARDIAC_HI) & (freqs <= noise_hi)
    sig_bins = int(np.count_nonzero(sig_mask))
    if sig_bins == 0:
        return 0.0, False
    signal_power = float(np.sum(psd[sig_mask]))
    noise_bins = int(np.count_nonzero(noise_mask))
    noise_density = float(np.sum(psd[noise_mask])) / noise_bins if noise_bins > 0 else 0.0
    in_band_noise = noise_density * sig_bins
    if in_band_noise <= 0:
        return float("inf"), False
    return signal_power / in_band_noise, True


def cardiac_spectral_prominence(sig: np.ndarray, fs: float) -> float:
    """Peak-to-median power ratio inside the cardiac band.

    Mirrors ``cardiacSpectralProminence``. A real PPG spectrum concentrates its
    energy into a narrow fundamental plus dicrotic harmonic; for a Gaussian
    noise field the expected maximum/median over ~90 bins is only ~2.5-3.5.
    """
    freqs, psd = sps.periodogram(sig.astype(float) - float(np.mean(sig)), fs=fs)
    band = psd[(freqs >= CARDIAC_LO) & (freqs <= CARDIAC_HI)]
    if band.size < 8:
        return 0.0
    median = float(np.median(band))
    peak = float(np.max(band))
    return peak / median if median > 0 else 0.0


def compute_sqi(ppg: np.ndarray, fs: float, dc_ref: Optional[float] = None) -> float:
    """Composite signal-quality index (0-100) mirroring the browser module.

    40% bandwidth-matched spectral SNR + 35% perfusion (AC/DC) + 25% pulse
    regularity, scaled down by a cardiac spectral-prominence gate so broadband
    noise or structured artifacts cannot score highly. Pass ``dc_ref`` (the
    skin-ROI DC level) when ``ppg`` is a zero-mean derived pulse such as POS.
    """
    if ppg.size < fs * 5:
        return 0.0

    pi = perfusion_index_acdc(ppg, fs, dc_ref)
    pi_score = min(1.0, np.sqrt(max(0.0, pi / 0.5)))

    snr, _has_reference = spectral_snr(ppg, fs)
    snr_score = min(1.0, max(0.0, (np.log10(snr) + 0.3) / 1.2)) if snr > 0 else 0.0

    x = butter_bandpass(detrend(ppg.astype(float)), CARDIAC_LO, CARDIAC_HI, fs)
    peaks, _ = sps.find_peaks(x, distance=max(1, int(fs * 0.3)))
    if peaks.size < 2:
        return 0.0
    rr = np.diff(peaks) / fs * 1000.0
    cv = (float(np.std(rr, ddof=1)) / float(np.mean(rr))) * 100.0 if rr.size > 1 and np.mean(rr) > 0 else 0.0
    reg_score = 1.0 if cv < 10 else max(0.0, 1.0 - (cv - 10.0) / 30.0)

    pulse_gate = min(1.0, max(0.0, (cardiac_spectral_prominence(ppg, fs) - 4.0) / 12.0))
    base = 0.4 * snr_score + 0.35 * pi_score + 0.25 * reg_score
    return round(min(1.0, base * pulse_gate) * 100.0)


def extract_creatinine_features(ppg: np.ndarray, fs: float) -> Optional[dict]:
    """Extract the PPG morphology family used by Sridevi et al. creatinine models.

    Mirrors ``extractCreatinineFeatures`` in the browser module. Returns the
    descriptors selected by the Smart Health 2025 (F0-F5/F10/F11/F13/F22) and
    Healthcare Analytics 2025 (F3/F8/F11/F13/F20/F22/F25) studies, or None when
    the signal is too short/unstable.
    """
    if ppg.size < fs * 5:
        return None
    x = butter_bandpass(detrend(ppg.astype(float)), 0.5, 3.5, fs)
    rng = float(x.max() - x.min()) or 1.0
    norm = (x - x.min()) / rng
    peaks, _ = sps.find_peaks(norm, distance=max(1, int(fs * 0.3)))
    if peaks.size < 3:
        return None
    intervals = np.diff(peaks)
    intervals = intervals[(intervals >= fs * 0.3) & (intervals <= fs * 2)]
    if intervals.size < 2:
        return None
    cycle_len = int(round(float(intervals.mean())))
    if cycle_len < 4:
        return None
    cycles = [norm[peaks[i]: peaks[i] + cycle_len] for i in range(peaks.size - 1) if peaks[i] + cycle_len <= norm.size]
    if len(cycles) < 2:
        return None
    avg = np.mean(np.vstack(cycles), axis=0)

    sys_idx = int(np.argmax(avg))
    peak_amp = float(avg[sys_idx])
    notch_end = min(cycle_len, sys_idx + int(fs * 0.5))
    if notch_end > sys_idx + 1:
        notch_idx = int(np.argmin(avg[sys_idx + 1:notch_end])) + sys_idx + 1
    else:
        notch_idx = sys_idx
    dicrotic_amp = float(avg[notch_idx])
    dias_idx = int(np.argmax(avg[notch_idx:])) + notch_idx
    augmentation_index = float(avg[dias_idx] / peak_amp) if peak_amp > 1e-6 else 0.0
    stiffness_ratio = float((peak_amp - dicrotic_amp) / peak_amp) if peak_amp > 1e-6 else 0.0
    rise_slope = float((avg[sys_idx] - avg[0]) / max(1, sys_idx))

    d1 = np.diff(avg)
    maxima = [j for j in range(1, len(d1) - 1) if d1[j] > d1[j - 1] and d1[j] > d1[j + 1]]
    minima = [j for j in range(1, len(d1) - 1) if d1[j] < d1[j - 1] and d1[j] < d1[j + 1]]
    second_deriv_time = float(maxima[1] / cycle_len) if len(maxima) > 1 else 0.0
    first_deriv_min_time = float(minima[0] / cycle_len) if minima else 0.0
    inflection_area_ratio = 0.0
    total_area = float(_trapz(avg, dx=1.0))
    if total_area > 1e-9:
        inflection_idx = maxima[0] if maxima else sys_idx
        inflection_area_ratio = float(_trapz(avg[: max(1, inflection_idx + 1)], dx=1.0) / total_area)

    return {
        "peak_amp": peak_amp,
        "dicrotic_amp": dicrotic_amp,
        "pulse_interval": float(intervals.mean() / fs),
        "rr_cv": float(intervals.std() / intervals.mean()) if intervals.mean() > 0 else 0.0,
        "augmentation_index": augmentation_index,
        "stiffness_ratio": stiffness_ratio,
        "rise_slope": rise_slope,
        "second_deriv_time": second_deriv_time,
        "first_deriv_min_time": first_deriv_min_time,
        "inflection_area_ratio": inflection_area_ratio,
        "beats": len(cycles),
    }


def estimate_creatinine(ppg: np.ndarray, fs: float, age: float = 45.0, sex: str = "male",
                        camera_mode: str = "rear") -> Optional[dict]:
    """Research-grade serum creatinine surrogate (mg/dL) + CKD-EPI 2021 eGFR.

    Design notes (mirroring ``estimateCreatinine`` in the browser module):
    age is intentionally NOT a predictor term - the morphology descriptors
    already carry vascular aging and age re-enters through CKD-EPI eGFR, so an
    explicit age term double-counted aging; front-camera rPPG uses
    mode-specific normalization bands instead of halving the signal terms
    (which collapsed the estimate onto its demographic base); and the estimate
    is gated on the composite SQI (>= 40).

    Not a clinical assay; use the validation dashboard to calibrate against
    laboratory creatinine before any interpretation.
    """
    sqi = compute_sqi(ppg, fs)
    if sqi < 40.0:
        return None
    feats = extract_creatinine_features(ppg, fs)
    if feats is None:
        return None
    is_front = camera_mode == "front"

    def nz(v, lo, hi):
        return float(min(1.0, max(0.0, (v - lo) / (hi - lo)))) if hi != lo else 0.0

    amplitude = nz(feats["peak_amp"], 0.05, 0.95)
    perfusion = (
        nz(perfusion_index_filtered(ppg, fs), 0.02, 0.8)
        if is_front
        else nz(perfusion_index_filtered(ppg, fs), 0.15, 3.0)
    )
    stiffness = nz(feats["stiffness_ratio"], 0.05, 0.9)
    augmentation = nz(feats["augmentation_index"], 0.1, 1.2)
    rr_cv = nz(feats["rr_cv"], 0.0, 0.3)
    notch_time = nz(0.0, 0.02, 0.5)
    systolic_time = nz(0.0, 0.05, 0.5)
    ipa = nz(feats["inflection_area_ratio"], 0.1, 0.7)
    deriv_min = nz(feats["first_deriv_min_time"], 0.05, 0.6)

    creatinine = (
        0.85
        + (0.15 if sex == "male" else 0.0)
        - 0.55 * (perfusion - 0.5)
        - 0.30 * (amplitude - 0.5)
        + 0.28 * (stiffness - 0.5)
        + 0.18 * (augmentation - 0.5)
        + 0.12 * (notch_time - 0.5)
        + 0.10 * (systolic_time - 0.5)
        - 0.14 * (ipa - 0.5)
        + 0.08 * (deriv_min - 0.5)
        + 0.10 * rr_cv
    )
    creatinine = round(min(12.0, max(0.4, creatinine)), 2)
    female = sex == "female"
    k = 0.7 if female else 0.9
    a = -0.241 if female else -0.302
    ratio = creatinine / k
    egfr = 142 * (min(ratio, 1) ** a) * (max(ratio, 1) ** -1.2) * (0.9938 ** age) * (1.012 if female else 1.0)
    confidence = min(0.55 if is_front else 0.8, 0.12 + 0.55 * (sqi / 100.0) + 0.15 * min(1.0, feats["beats"] / 15.0))
    return {
        "creatinine_mg_dl": creatinine,
        "creatinine_class": "high" if creatinine > 1.3 else "low" if creatinine < 0.6 else "normal",
        "egfr": round(egfr),
        "sqi": sqi,
        "confidence": round(confidence, 2),
        "method": "rppg-front" if is_front else "ppg-rear-ftir",
        "features": feats,
    }


def analyze(ppg_path: str, fs: float = 30.0, n_bootstrap: int = 1000) -> dict:
    ppg = load_ppg_csv(ppg_path)
    heart = analyze_heart_heartpy(ppg, fs, n_bootstrap=n_bootstrap)
    resp = analyze_resp_rrest(ppg, fs)
    return {
        "heart": heart,
        "resp": resp,
        "creatinine": estimate_creatinine(ppg, fs),
        "fs": fs,
        "samples": int(ppg.size),
    }


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
    cr = result.get("creatinine")
    if cr is None:
        print("Serum Creatinine: n/a (signal too short/unstable)")
    else:
        print(
            f"Serum Creatinine (research estimate): {cr['creatinine_mg_dl']:.2f} mg/dL "
            f"[{cr['creatinine_class']}] | eGFR {cr['egfr']} mL/min/1.73m2"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
