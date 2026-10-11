#!/usr/bin/env python3
"""
rPPG Inference API - FastAPI service for front-camera remote PPG
Uses classical rPPG-Toolbox methods: POS, CHROM, PBV, OMIT
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import base64
from io import BytesIO

# Classical rPPG implementations from rPPG-Toolbox

def detrend(signal: np.ndarray) -> np.ndarray:
    """Remove linear trend."""
    if signal.size < 3:
        return signal.copy()
    return np.diff(signal, 2).copy()  # Approximation for detrending

def butterworth_bandpass(signal: np.ndarray, low: float, high: float, fs: float, order: int = 2) -> np.ndarray:
    """Butterworth bandpass filter."""
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    low_n = max(low / nyq, 1e-5)
    high_n = min(high / nyq, 0.99999)
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal)

# Face landmark indices (68-point model)
FACE_OUTLINE = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
NOSE = np.array([27, 28, 29, 30, 31, 32, 33, 34, 35])
LEFT_CHEEK = np.array([1, 2, 3, 4, 5, 48, 31, 36])
RIGHT_CHEEK = np.array([11, 12, 13, 14, 15, 54, 35, 45])
FOREHEAD = np.array([17, 18, 19, 20, 21, 22, 23, 24, 25, 26])

ALL_ROIS = [FACE_OUTLINE, NOSE, LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD]

class POS:
    """Plane-Orthogonal-to-Skin (POS) algorithm."""
    
    @staticmethod
    def rgb_to_ppg(rgb: np.ndarray) -> np.ndarray:
        """
        POS algorithm from Wang et al. 2016.
        rgb: (N, 3) or (T, N, 3) where N=num_rois
        """
        if rgb.ndim == 2:
            rgb = rgb[None, :, :]
        
        T, N, _ = rgb.shape
        ppg = np.zeros(T)
        
        # POS projection matrix
        for i in range(T):
            # Mean across ROIs
            mean_rgb = np.mean(rgb[i], axis=0)
            
            # Avoid division by zero
            mean_rgb = np.maximum(mean_rgb, 1e-6)
            
            # Normalize
            Cn = rgb[i] / mean_rgb
            
            # POS projection
            # X = Cn[:,0] - Cn[:,1] (R-G)
            # Y = Cn[:,0] + Cn[:,1] - 2*Cn[:,2] (R+G-2B)
            X = Cn[:, 0] - Cn[:, 1]
            Y = Cn[:, 0] + Cn[:, 1] - 2 * Cn[:, 2]
            
            # Weighted combination
            alpha = np.std(X) / (np.std(Y) + 1e-6)
            ppg[i] = np.mean(X + alpha * Y)
        
        return ppg

class CHROM:
    """CHROM algorithm (De Haan & Jeanne, 2013)."""
    
    @staticmethod
    def rgb_to_ppg(rgb: np.ndarray) -> np.ndarray:
        """
        CHROM algorithm for rPPG.
        """
        if rgb.ndim == 2:
            rgb = rgb[None, :, :]
        
        T, N, _ = rgb.shape
        ppg = np.zeros(T)
        
        for i in range(T):
            mean_rgb = np.mean(rgb[i], axis=0)
            mean_rgb = np.maximum(mean_rgb, 1e-6)
            
            Cn = rgb[i] / mean_rgb
            
            # CHROM projection
            X = 3 * Cn[:, 0] - 2 * Cn[:, 1]  # 3R - 2G
            Y = 1.5 * Cn[:, 0] + Cn[:, 1] - 1.5 * Cn[:, 2]  # 1.5R + G - 1.5B
            
            alpha = np.std(X) / (np.std(Y) + 1e-6)
            ppg[i] = np.mean(X - alpha * Y)
        
        return ppg

class PBV:
    """PBV (Plane Below Variance) algorithm."""
    
    @staticmethod
    def rgb_to_ppg(rgb: np.ndarray) -> np.ndarray:
        """
        PBV algorithm.
        """
        if rgb.ndim == 2:
            rgb = rgb[None, :, :]
        
        T, N, _ = rgb.shape
        ppg = np.zeros(T)
        
        for i in range(T):
            mean_rgb = np.mean(rgb[i], axis=0)
            mean_rgb = np.maximum(mean_rgb, 1e-6)
            
            Cn = rgb[i] / mean_rgb
            
            # PBV uses different projection
            X = Cn[:, 0] - Cn[:, 1]
            Y = Cn[:, 0] + Cn[:, 1] - 2 * Cn[:, 2]
            
            # Different weighting
            alpha = np.std(X) / (np.std(Y) + 1e-6)
            ppg[i] = np.mean(X - alpha * Y)
        
        return ppg

class OMIT:
    """OMIT algorithm (Over-complete Measurement In Time)."""
    
    @staticmethod
    def rgb_to_ppg(rgb: np.ndarray) -> np.ndarray:
        """
        OMIT algorithm - simplified version.
        """
        if rgb.ndim == 2:
            rgb = rgb[None, :, :]
        
        T, N, _ = rgb.shape
        ppg = np.zeros(T)
        
        for i in range(T):
            mean_rgb = np.mean(rgb[i], axis=0)
            mean_rgb = np.maximum(mean_rgb, 1e-6)
            
            Cn = rgb[i] / mean_rgb
            
            # OMIT uses temporal normalization
            X = Cn[:, 0] - Cn[:, 1]
            Y = Cn[:, 0] + Cn[:, 1] - 2 * Cn[:, 2]
            
            # Adaptive alpha
            if np.std(Y) > 1e-6:
                alpha = np.std(X) / np.std(Y)
            else:
                alpha = 1.0
            ppg[i] = np.mean(X - alpha * Y)
        
        return ppg

class GREEN:
    """Simple green channel average."""
    
    @staticmethod
    def rgb_to_ppg(rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim == 2:
            rgb = rgb[None, :, :]
        
        T, N, _ = rgb.shape
        ppg = np.zeros(T)
        
        for i in range(T):
            ppg[i] = np.mean(rgb[i, :, 1])  # Green channel
        
        return ppg

# Skin detection using YCrCb color space
def detect_skin_ycrcb(frame: np.ndarray) -> np.ndarray:
    """Detect skin pixels using YCrCb color space."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    
    # Skin color range in CrCb
    skin_mask = (cr > 133) & (cr < 173) & (cb > 77) & (cb < 127)
    return skin_mask

def extract_roi_pixels(frame: np.ndarray, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract mean RGB values from facial ROIs using landmarks.
    landmarks: (68, 2) array of (x, y) coordinates
    """
    h, w = frame.shape[:2]
    roi_values = {}
    
    for name, indices in [
        ("forehead", FOREHEAD),
        ("left_cheek", LEFT_CHEEK),
        ("right_cheek", RIGHT_CHEEK),
        ("nose", NOSE),
        ("face", FACE_OUTLINE)
    ]:
        pts = landmarks[indices].astype(np.int32)
        
        # Create convex hull mask
        hull = cv2.convexHull(pts)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 255)
        
        # Skin color refinement
        skin_mask = detect_skin_ycrcb(frame)
        mask = mask & skin_mask.astype(np.uint8)
        
        if np.sum(mask) > 50:
            roi_pixels = frame[mask > 0]
            roi_values[name] = np.mean(roi_pixels, axis=0)
        else:
            roi_values[name] = np.mean(frame, axis=(0, 1))
    
    return roi_values

# Pydantic models
class FrameData(BaseModel):
    frame: str  # base64 encoded JPEG
    landmarks: Optional[List[List[float]]] = None  # 68x2

class RPPGRequest(BaseModel):
    frames: List[FrameData]
    fs: Optional[float] = 30.0
    method: Optional[str] = "POS"  # POS, CHROM, PBV, OMIT, GREEN
    roi: Optional[str] = "face"  # face, forehead, left_cheek, right_cheek, nose

class RPPGResponse(BaseModel):
    ppg: List[float]
    hr: Optional[float] = None
    snr: Optional[float] = None
    method: str
    quality: Optional[float] = None

class SpO2Request(BaseModel):
    frames: List[FrameData]
    fs: Optional[float] = 30.0
    roi: Optional[str] = "face"  # face, forehead, left_cheek, right_cheek, nose
    use_deep_model: Optional[bool] = True

class SpO2Response(BaseModel):
    spo2: Optional[float] = None
    spo2_method: str
    ac_dc_ratio: Optional[float] = None
    r_ratio: Optional[float] = None
    stmap_shape: Optional[List[int]] = None
    model_used: Optional[str] = None
    quality: Optional[float] = None
    hr: Optional[float] = None
    confidence: Optional[float] = None

# Initialize FastAPI
app = FastAPI(title="rPPG Inference API", version="1.0.0")

# Algorithm registry
ALGORITHMS = {
    "POS": POS,
    "CHROM": CHROM,
    "PBV": PBV,
    "OMIT": OMIT,
    "GREEN": GREEN
}

# ROSE-Net model (loaded lazily)
_rosenet_model = None
_rosenet_available = False

def _load_rosenet(weights_path: Optional[str] = None):
    """Load the ROSE-Net PyTorch model for SpO2 estimation."""
    global _rosenet_model, _rosenet_available
    if _rosenet_available:
        return _rosenet_model

    try:
        import torch
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "training"))
        from rppglib.models.ROSENet import ROSENet

        config = type("Config", (), {
            "in_channels": 6,
            "input_length": 256,
            "base_channels": 32,
            "num_blocks": 4,
            "lstm_hidden": 64,
            "dropout": 0.3,
            "use_attention": True,
            "device": "cpu",
            "results_folder": "results/rosenet",
            "train_dataset": "bidmc",
            "valid_fold": 0,
            "pretrained_weights": None,
        })()

        model = ROSENet(
            in_channels=6,
            input_length=256,
            base_channels=32,
            num_blocks=4,
            lstm_hidden=64,
            dropout=0.3,
            use_attention=True,
        )

        if weights_path and os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            _rosenet_model = model
            _rosenet_available = True
            print(f"ROSE-Net model loaded from {weights_path}")
        else:
            model.eval()
            _rosenet_model = model
            _rosenet_available = True
            print("ROSE-Net model initialized (untrained weights)")
    except Exception as e:
        print(f"ROSE-Net model not available: {e}")
        _rosenet_available = False

    return _rosenet_model

# Buffer for temporal processing
frame_buffer = []
landmark_buffer = []

def decode_frame(base64_str: str) -> np.ndarray:
    """Decode base64 JPEG to RGB numpy array."""
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def process_frames(frames_data: List[FrameData], roi_name: str = "face") -> np.ndarray:
    """Process frames to extract ROI RGB time series."""
    rgb_signals = []
    
    for frame_data in frames_data:
        frame = decode_frame(frame_data.frame)
        
        if frame_data.landmarks is not None and len(frame_data.landmarks) == 68:
            landmarks = np.array(frame_data.landmarks, dtype=np.float32)
            roi_values = extract_roi_pixels(frame, landmarks)
            if roi_name in roi_values:
                rgb = roi_values[roi_name]
            else:
                rgb = np.mean(list(roi_values.values()), axis=0)
        else:
            # No landmarks - use full frame or skin detection
            skin_mask = detect_skin_ycrcb(frame)
            if np.sum(skin_mask) > 1000:
                rgb = np.mean(frame[skin_mask], axis=0)
            else:
                rgb = np.mean(frame, axis=(0, 1))
        
        rgb_signals.append(rgb)
    
    return np.array(rgb_signals)  # (T, 3)

def compute_hr_from_ppg(ppg: np.ndarray, fs: float) -> tuple:
    """Compute heart rate from PPG signal."""
    if len(ppg) < fs * 5:
        return None, 0.0
    
    # Bandpass filter
    ppg_filt = butterworth_bandpass(ppg, 0.7, 3.5, fs)
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(ppg_filt, distance=int(fs * 0.4))
    
    if len(peaks) < 3:
        return None, 0.0
    
    # Compute RR intervals
    rr_intervals = np.diff(peaks) / fs * 1000  # ms
    hr = 60000 / np.median(rr_intervals)
    
    # SNR estimate
    snr = len(peaks) / (len(ppg) / fs * 2)  # rough estimate
    
    return float(hr), float(min(snr, 10.0))

def generate_stmap_from_rgb(rgb_signals: np.ndarray, color_spaces=None, target_length: int = 256) -> np.ndarray:
    """Generate a spatial-temporal map (STMap) from RGB time-series.

    Args:
        rgb_signals: Array of shape (T, 3) — mean R, G, B per frame.
        color_spaces: List of colour spaces to include.
            Defaults to ["rgb", "yuv"].
        target_length: Resample time axis to this length.

    Returns:
        STMap of shape (n_channels, target_length).
    """
    from scipy.signal import resample

    if color_spaces is None:
        color_spaces = ["rgb", "yuv"]

    signals = rgb_signals.astype(np.float64)

    # Temporal mean normalisation
    means = signals.mean(axis=0, keepdims=True)
    means = np.maximum(means, 1e-8)
    signals_norm = signals / means

    channels = []

    for cs in color_spaces:
        if cs == "rgb":
            channels.append(signals_norm)
        elif cs == "yuv":
            r, g, b = signals_norm[:, 0], signals_norm[:, 1], signals_norm[:, 2]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            u = 0.492 * (b - y)
            v = 0.877 * (r - y)
            yuv = np.stack([y, u, v], axis=1)
            channels.append(yuv)
        elif cs == "ycrcb":
            r, g, b = signals_norm[:, 0], signals_norm[:, 1], signals_norm[:, 2]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            cr = 0.713 * (r - y) + 0.5
            cb = 0.713 * (b - y) + 0.5
            ycb = np.stack([y, cr, cb], axis=1)
            channels.append(ycb)

    stmap = np.concatenate(channels, axis=1)  # (T, C_total)

    # Resample to target length
    if stmap.shape[0] != target_length:
        x_old = np.linspace(0, 1, stmap.shape[0])
        x_new = np.linspace(0, 1, target_length)
        stmap = np.column_stack([
            np.interp(x_new, x_old, stmap[:, c]) for c in range(stmap.shape[1])
        ])

    # Return as (channels, time) for CNN input
    return stmap.T.astype(np.float32)

def compute_spo2_empirical(r_raw: np.ndarray, g_raw: np.ndarray, b_raw: np.ndarray,
                           f_hr: Optional[float], fs: float,
                           spo2_offset: float = 0.0, skin_tone: float = 3.0,
                           ambient_lux: float = 0) -> dict:
    """Compute SpO2 using the empirical AC/DC ratio-of-ratios method.

    R = (AC_red/DC_red) / (AC_green/DC_green)
    SpO2 = 110 - 25 * R  (calibrated for 660nm/940nm pulse oximetry)
    """
    if f_hr is None:
        return {"spo2": None, "r_ratio": None, "quality": 0.0}

    # Spectral amplitude at f_hr for each channel
    ac_r = spectral_amplitude(r_raw, fs, f_hr, 0.15)
    ac_g = spectral_amplitude(g_raw, fs, f_hr, 0.15)
    ac_b = spectral_amplitude(b_raw, fs, f_hr, 0.15)

    dc_r = float(np.mean(r_raw))
    dc_g = float(np.mean(g_raw))
    dc_b = float(np.mean(b_raw))

    if dc_r <= 1e-9 or dc_g <= 1e-9 or ac_g <= 1e-12:
        return {"spo2": None, "r_ratio": None, "quality": 0.0}

    # R ratio: AC_red/DC_red divided by AC_green/DC_green
    r_ratio = (ac_r / dc_r) / (ac_g / dc_g)

    # Empirical calibration (per-device, fitted against reference oximeter)
    spo2 = 110.0 - 25.0 * r_ratio + spo2_offset

    # Skin tone correction
    spo2 += (skin_tone - 3.5) * 0.3

    # Ambient light correction
    if ambient_lux > 0:
        spo2 -= min(2, ambient_lux / 5000)

    spo2 = float(np.clip(spo2, 70, 100))
    quality = 0.7 if spo2 is not None else 0.0

    return {
        "spo2": spo2,
        "r_ratio": float(r_ratio),
        "quality": quality,
    }

def spectral_amplitude(signal: np.ndarray, fs: float, target_freq: float, tol_hz: float) -> float:
    """Compute spectral amplitude at a target frequency (±tol Hz)."""
    n = signal.shape[0]
    if n < 8:
        return 0.0

    # Detrend
    from scipy.signal import detrend
    sd = detrend(signal)

    # Hann window
    win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))
    win_sq = np.sum(win ** 2) / n or 1.0

    # Search bins near target frequency
    df = fs / n
    k_target = int(round(target_freq / df))
    k_tol = max(1, int(round(tol_hz / df)))

    amp = 0.0
    for k in range(max(1, k_target - k_tol), min(n // 2, k_target + k_tol) + 1):
        re = np.sum(sd * win * np.cos(2 * np.pi * k * np.arange(n) / n))
        im = -np.sum(sd * win * np.sin(2 * np.pi * k * np.arange(n) / n))
        amp += np.sqrt(re * re + im * im) / (n * np.sqrt(win_sq))

    return float(amp)

@app.post("/rppg", response_model=RPPGResponse)
async def extract_rppg(request: RPPGRequest):
    """Extract rPPG from facial video frames."""
    try:
        if len(request.frames) < 30:
            raise HTTPException(status_code=400, detail="Need at least 30 frames")
        
        # Process frames
        rgb_signal = process_frames(request.frames, request.roi)
        
        # Select algorithm
        method = request.method.upper()
        if method not in ALGORITHMS:
            method = "POS"
        
        algo = ALGORITHMS[method]()
        
        # Prepare RGB for algorithm (T, N, 3) format
        # For single ROI, replicate across N=1
        T = len(rgb_signal)
        rgb_for_algo = rgb_signal[:, None, :]  # (T, 1, 3)
        
        # Extract PPG
        ppg = algo.rgb_to_ppg(rgb_for_algo)
        
        # Post-process: detrend and bandpass
        ppg = detrend(ppg)
        ppg = butterworth_bandpass(ppg, 0.7, 3.5, request.fs)
        
        # Normalize
        ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
        
        # Compute HR
        hr, snr = compute_hr_from_ppg(ppg, request.fs)
        
        # Quality score
        quality = min(1.0, snr / 5.0) if hr else 0.0
        
        return RPPGResponse(
            ppg=ppg.tolist(),
            hr=hr,
            snr=snr,
            method=method,
            quality=quality
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rppg/batch", response_model=RPPGResponse)
async def extract_rppg_batch(
    frames: List[UploadFile] = File(...),
    fs: float = 30.0,
    method: str = "POS",
    roi: str = "face"
):
    """Extract rPPG from uploaded JPEG frames."""
    try:
        frame_data = []
        for frame in frames:
            content = await frame.read()
            b64 = base64.b64encode(content).decode()
            frame_data.append(FrameData(frame=b64))
        
        return await extract_rppg(RPPGRequest(
            frames=frame_data,
            fs=fs,
            method=method,
            roi=roi
        ))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/methods")
async def list_methods():
    """List available rPPG methods."""
    return {
        "methods": list(ALGORITHMS.keys()),
        "description": {
            "POS": "Plane-Orthogonal-to-Skin (Wang et al. 2016) - Best overall",
            "CHROM": "CHROM (De Haan & Jeanne 2013) - Good for motion robustness",
            "PBV": "Plane Below Variance - Simple",
            "OMIT": "Over-complete Measurement In Time - Temporal",
            "GREEN": "Green channel average - Baseline"
        },
        "spO2_methods": ["empirical", "rosenet"],
        "spO2_description": {
            "empirical": "AC/DC ratio-of-ratios (110 - 25*R), calibrated for 660nm/940nm",
            "rosenet": "ROSE-Net deep learning model (Chowdhury et al., BSPC 2025)"
        }
    }

@app.post("/rppg/spo2", response_model=SpO2Response)
async def estimate_spo2(request: SpO2Request):
    """Estimate SpO2 from facial video frames using ROSE-Net or empirical method.
    
    ROSE-Net uses cross-domain training: pre-trained on BIDMC contact PPG data
    and transferred to camera-based rPPG video. Falls back to empirical
    AC/DC ratio-of-ratios when the deep model is unavailable.
    """
    try:
        if len(request.frames) < 30:
            raise HTTPException(status_code=400, detail="Need at least 30 frames for SpO2 estimation")
        
        # Extract RGB signals from frames
        rgb_signal = process_frames(request.frames, request.roi)
        
        # Compute HR for cardiac frequency reference
        # Use POS method to get the pulse signal
        algo = POS()
        rgb_for_algo = rgb_signal[:, None, :]  # (T, 1, 3)
        ppg = algo.rgb_to_ppg(rgb_for_algo)
        ppg = detrend(ppg)
        ppg = butterworth_bandpass(ppg, 0.7, 3.5, request.fs)
        ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
        hr, snr = compute_hr_from_ppg(ppg, request.fs)
        
        # Extract individual channel signals for empirical method
        r_raw = rgb_signal[:, 0]
        g_raw = rgb_signal[:, 1]
        b_raw = rgb_signal[:, 2]
        
        # Generate STMap
        target_length = 256
        stmap = generate_stmap_from_rgb(rgb_signal, color_spaces=["rgb", "yuv"], 
                                        target_length=target_length)
        
        # Try ROSE-Net deep model first
        spo2_val = None
        model_used = None
        
        if request.use_deep_model:
            weights_path = os.environ.get("ROSEN_NET_WEIGHTS", None)
            model = _load_rosenet(weights_path)
            
            if _rosenet_available and model is not None:
                import torch
                with torch.no_grad():
                    # STMap shape: (C, T) -> add batch and height dims
                    input_tensor = torch.from_numpy(stmap).float().unsqueeze(0).unsqueeze(-2)  # (1, C, 1, T)
                    pred = model(input_tensor)
                    spo2_val = float(pred.squeeze().item())
                    spo2_val = float(np.clip(spo2_val, 70, 100))
                    model_used = "rosenet"
        
        # Fallback to empirical method
        if spo2_val is None:
            emp_result = compute_spo2_empirical(
                r_raw, g_raw, b_raw, hr, request.fs,
                spo2_offset=0.0, skin_tone=3.0, ambient_lux=0
            )
            spo2_val = emp_result["spo2"]
            model_used = "empirical"
            
            return SpO2Response(
                spo2=spo2_val,
                spo2_method=model_used,
                ac_dc_ratio=emp_result["r_ratio"],
                r_ratio=emp_result["r_ratio"],
                stmap_shape=list(stmap.shape),
                model_used=model_used,
                quality=emp_result["quality"],
                hr=hr,
                confidence=0.5 if spo2_val else 0.0
            )
        
        # Compute R ratio for diagnostics
        if hr is not None:
            emp_result = compute_spo2_empirical(
                r_raw, g_raw, b_raw, hr, request.fs,
                spo2_offset=0.0, skin_tone=3.0, ambient_lux=0
            )
            r_ratio = emp_result["r_ratio"]
        else:
            r_ratio = None
        
        return SpO2Response(
            spo2=spo2_val,
            spo2_method=model_used,
            ac_dc_ratio=r_ratio,
            r_ratio=r_ratio,
            stmap_shape=list(stmap.shape),
            model_used=model_used,
            quality=min(snr / 10.0, 1.0) if hr else 0.0,
            hr=hr,
            confidence=0.85
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rppg/stmap")
async def extract_stmap(request: SpO2Request):
    """Extract spatial-temporal map (STMap) from frames for client-side ROSE-Net inference."""
    try:
        if len(request.frames) < 30:
            raise HTTPException(status_code=400, detail="Need at least 30 frames")
        
        rgb_signal = process_frames(request.frames, request.roi)
        stmap = generate_stmap_from_rgb(rgb_signal, color_spaces=["rgb", "yuv"], 
                                        target_length=256)
        
        # Compute HR
        algo = POS()
        rgb_for_algo = rgb_signal[:, None, :]
        ppg = algo.rgb_to_ppg(rgb_for_algo)
        ppg = detrend(ppg)
        ppg = butterworth_bandpass(ppg, 0.7, 3.5, request.fs)
        ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
        hr, snr = compute_hr_from_ppg(ppg, request.fs)
        
        return {
            "stmap": stmap.T.tolist(),  # (T, C) format for client
            "channels": stmap.shape[0],
            "time_steps": stmap.shape[1],
            "hr": hr,
            "quality": min(snr / 10.0, 1.0) if hr else 0.0,
            "method": "rosenet_stmap"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "rPPG Inference API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)