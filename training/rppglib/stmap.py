"""
STMap (Spatial-Temporal Map) generation for ROSE-Net.

An STMap is a 2-D representation of the RGB (and optionally YUV / YCrCb)
colour intensities over time.  For a video of *T* frames, the mean ROI
colour in each frame is extracted, normalised by the temporal mean
(``Cn = C / mean(C)``), and arranged as columns of a 2-D array.

The resulting array has shape ``(T, C)`` (time × channels) for a single
colour space, or ``(C_total, T)`` when multiple spaces are stacked.

Cross-domain training bridge
-----------------------------
To pre-train ROSE-Net on contact PPG data (BIDMC), we synthesise RGB
signals from the PPG waveform by modelling how a reflective pulse-oximeter
LED at 660 nm (red) and 940 nm (infrared) transilluminates tissue.  This
creates STMaps that share the same AC/DC ratio structure as camera-based
rPPG, enabling domain transfer.
"""
import numpy as np
import cv2
from typing import Optional, Tuple, List


# ─────────────────────────────────────────────────────────────────────
# Colour-space conversions
# ─────────────────────────────────────────────────────────────────────

def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to YUV.  ``rgb`` has shape (T, 3) or (T, H, W, 3)."""
    if rgb.ndim == 2:
        rgb = rgb[:, None, :]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 0.492 * (b - y)
    v = 0.877 * (r - y)
    return np.stack([y, u, v], axis=-1).reshape(rgb.shape)


def rgb_to_ycrcb(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to YCrCb."""
    if rgb.ndim == 2:
        rgb = rgb[:, None, :]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = 0.713 * (r - y) + 0.5
    cb = 0.713 * (b - y) + 0.5
    return np.stack([y, cr, cb], axis=-1).reshape(rgb.shape)


def normalize_channels(signals: np.ndarray) -> np.ndarray:
    """Temporal mean-normalisation: ``Cn = C / mean(C)``."""
    means = signals.mean(axis=0, keepdims=True)
    means = np.maximum(means, 1e-8)
    return signals / means


# ─────────────────────────────────────────────────────────────────────
# STMap from RGB time-series
# ─────────────────────────────────────────────────────────────────────

def generate_stmap(
    rgb_signals: np.ndarray,
    color_spaces: List[str] = ["rgb"],
    normalize: bool = True,
    target_length: Optional[int] = None,
) -> np.ndarray:
    """Build a spatial-temporal map from per-frame RGB means.

    Args:
        rgb_signals: Array of shape ``(T, 3)`` — mean R, G, B per frame.
        color_spaces: Which colour spaces to include.
            Supported: ``"rgb"``, ``"yuv"``, ``"ycrcb"``.
        normalize:    Apply temporal mean-normalisation (``Cn = C / mean(C)``).
        target_length: If given, resample time axis to this length.

    Returns:
        STMap of shape ``(n_channels, T)`` or ``(n_channels, T)``
        depending on ``target_length``.
    """
    if rgb_signals.ndim == 1:
        rgb_signals = rgb_signals.reshape(-1, 1)
    if rgb_signals.ndim == 2 and rgb_signals.shape[1] != 3:
        # Single-channel input (e.g. PPG waveform)
        rgb_signals = np.tile(rgb_signals, (1, 3))

    signals = rgb_signals.astype(np.float64)

    if normalize:
        signals = normalize_channels(signals)

    channels = []
    for cs in color_spaces:
        if cs == "rgb":
            channels.append(signals)
        elif cs == "yuv":
            yuv = rgb_to_yuv(signals)
            if normalize:
                yuv = normalize_channels(yuv)
            channels.append(yuv)
        elif cs == "ycrcb":
            ycb = rgb_to_ycrcb(signals)
            if normalize:
                ycb = normalize_channels(ycb)
            channels.append(ycb)

    stmap = np.concatenate(channels, axis=1)  # (T, C_total)

    if target_length is not None:
        stmap = resample_time(stmap, target_length)

    # Return as (channels, time) for CNN input
    return stmap.T.astype(np.float32)


def generate_stmap_from_video(
    video: np.ndarray,
    landmarks: Optional[np.ndarray] = None,
    roi_indices: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Generate STMap from raw video frames.

    Args:
        video:    Array of shape ``(T, H, W, 3)`` — uint8 frames.
        landmarks: Optional ``(T, N, 2)`` landmark array.
        roi_indices: Optional face-landmark indices defining the ROI.
        **kwargs:  Forwarded to ``generate_stmap``.

    Returns:
        STMap of shape ``(C, T)``.
    """
    if landmarks is not None and roi_indices is not None:
        rgb_means = _extract_roi_means(video, landmarks, roi_indices)
    else:
        rgb_means = _extract_full_frame_means(video)

    return generate_stmap(rgb_means, **kwargs)


def _extract_full_frame_means(video: np.ndarray) -> np.ndarray:
    """Compute mean R, G, B per frame."""
    T = video.shape[0]
    rgb = np.zeros((T, 3), dtype=np.float64)
    for i in range(T):
        frame = video[i].astype(np.float64)
        rgb[i] = frame.reshape(-1, 3).mean(axis=0)
    return rgb


def _extract_roi_means(
    video: np.ndarray, landmarks: np.ndarray, roi_indices: np.ndarray
) -> np.ndarray:
    """Compute mean RGB inside a polygonal ROI defined by landmarks."""
    T, H, W, _ = video.shape
    rgb = np.zeros((T, 3), dtype=np.float64)

    for i in range(T):
        pts = landmarks[i][roi_indices].astype(np.int32)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        masked = video[i][mask > 0]
        if len(masked) > 0:
            rgb[i] = masked.mean(axis=0)
        else:
            rgb[i] = video[i].reshape(-1, 3).mean(axis=0)

    return rgb


def resample_time(data: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample the time axis to ``target_length`` samples."""
    n = data.shape[0]
    if n == target_length:
        return data
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, data)


# ─────────────────────────────────────────────────────────────────────
# Cross-domain synthesis: PPG → RGB
# ─────────────────────────────────────────────────────────────────────

# Camera red and blue spectral response peaks (nm)
CAM_RED_PEAK = 620
CAM_BLUE_PEAK = 460

# Pulse oximeter LED wavelengths (nm)
PURE_RED_LED = 660   # ~660 nm for SpO2 red
PURE_IR_LED = 940    # ~940 nm for SpO2 infrared

# Oxy/deoxyhemoglobin extinction coefficients at key wavelengths (cm^-1 M^-1)
# From known literature values.
EPS_HBO2_RED = 76     # ε_HbO2 at 660 nm
EPS_HB_RED = 1637     # ε_Hb  at 660 nm
EPS_HBO2_IR = 600     # ε_HbO2 at 940 nm  (~400, approximated for camera IR proxy)
EPS_HB_IR = 800       # ε_Hb  at 940 nm  (~1100, approximated)

# Camera sensor QE at wavelengths (approximate)
CAMERA_QE_RED = 0.5
CAMERA_QE_IR = 0.15  # cameras have very low sensitivity at 940 nm


def synthesize_rgb_from_ppg(
    ppg: np.ndarray,
    spo2: float,
    fs: float = 30.0,
    heart_rate: Optional[float] = None,
    skin_tone: float = 3.0,
    ambient_lux: float = 500.0,
) -> np.ndarray:
    """Synthesise RGB signals from a contact PPG waveform and reference SpO2.

    Models the Beer-Lambert absorption of oxy- and deoxyhemoglobin in the
    red (620–700 nm) and "infrared" (900–940 nm) bands, then maps the
    transmitted intensities into the camera R / G / B channels.

    This creates synthetic STMaps that share the same AC/DC ratio structure
    as real facial rPPG, enabling cross-domain pre-training of ROSE-Net
    on the BIDMC contact-PPG dataset.

    Args:
        ppg:       1-D PPG waveform (arbitrary units, already normalised).
        spo2:      Reference SpO2 (%) from the oximeter (70–100).
        fs:        Sampling rate of ``ppg``.
        heart_rate: Optional HR in BPM (for cardiac-frequency band definition).
        skin_tone: Fitzpatrick skin type (1–6); modulates baseline DC.
        ambient_lux: Ambient light level; modulates noise floor.

    Returns:
        RGB signals of shape ``(T, 3)``.
    """
    n = len(ppg)
    t = np.arange(n) / fs

    # Normalise PPG to zero mean, unit std pulse component
    ppg_norm = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)

    # AC component (pulsatile) — extract via bandpass around HR
    if heart_rate is not None and heart_rate > 0:
        hr_hz = heart_rate / 60.0
        low = max(0.5, hr_hz - 1.0)
        high = min(fs / 2 - 0.1, hr_hz + 2.0)
    else:
        low, high = 0.5, 3.5

    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    ac = filtfilt(
        *butter(2, [max(low / nyq, 1e-4), min(high / nyq, 0.999)], btype="band"),
        ppg_norm,
    )
    dc = np.mean(ppg)

    # SpO2-dependent absorption ratio (R)
    # Higher SpO2 → lower AC_red/DC ratio relative to AC_IR/DC
    # R ≈ a + b * SpO2  (inverse of the empirical calibration)
    r_ratio = 1.4 - 0.012 * spo2  # R decreases as SpO2 increases

    # Fractional oxyhemoglobin
    spo2_frac = spo2 / 100.0

    # Extinction coefficients for Hb and HbO2 at the two wavelengths
    # μ = ε · c · L  (Beer-Lambert)
    # AC ∝ μ_ppg · (ε_HbO2 - ε_Hb)  for the pulsatile component
    mu_ac_red = abs(
        spo2_frac * EPS_HBO2_RED +
        (1 - spo2_frac) * EPS_HB_RED
    )
    mu_ac_ir = abs(
        spo2_frac * EPS_HBO2_IR +
        (1 - spo2_frac) * EPS_HB_IR
    )

    # DC level (skin tone modulates baseline reflectance)
    skin_factor = 0.8 + 0.1 * skin_tone  # darker skin → lower reflectance

    # Simulate camera response
    # Red channel: primarily 660 nm LED, some ambient
    red_dc = 120.0 * skin_factor * (1 + ambient_lux / 5000)
    # Green channel: reflected ambient + minimal pulse
    green_dc = 130.0 * skin_factor
    # Blue channel: proxy for IR (cameras have low IR sensitivity)
    blue_dc = 80.0 * skin_factor * CAMERA_QE_IR / CAMERA_QE_RED

    # AC amplitudes
    red_ac_amp = 0.5 * mu_ac_red * abs(ac)
    ir_ac_amp = 0.5 * mu_ac_ir * abs(ac)
    green_ac_amp = 0.3 * mu_ac_red * abs(ac)  # green carries pulse info too

    # Synthesize RGB
    r = red_dc + red_ac_amp * ac
    g = green_dc + green_ac_amp * ac
    b = blue_dc + ir_ac_amp * ac  # blue proxies IR response

    # Add realistic sensor noise (shot noise + read noise)
    shot_noise_std = np.sqrt(np.maximum(np.stack([r, g, b], axis=-1), 0)) * 0.02
    read_noise = np.random.normal(0, 1.5, size=(n, 3))
    rgb = np.stack([r, g, b], axis=-1) + shot_noise_std + read_noise

    return rgb.astype(np.float32)


def generate_synthetic_stmap(
    ppg: np.ndarray,
    spo2: float,
    fs: float = 30.0,
    heart_rate: Optional[float] = None,
    target_length: int = 256,
    color_spaces: List[str] = ["rgb", "yuv"],
) -> Tuple[np.ndarray, float]:
    """Generate a synthetic STMap from a PPG waveform and reference SpO2.

    This is the cross-domain bridge: contact PPG → synthetic RGB → STMap.

    Returns:
        (stmap, spo2) where stmap has shape ``(C, target_length)``.
    """
    rgb = synthesize_rgb_from_ppg(ppg, spo2, fs, heart_rate)
    stmap = generate_stmap(rgb, color_spaces=color_spaces, target_length=target_length)
    return stmap, spo2
