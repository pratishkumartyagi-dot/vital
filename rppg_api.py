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
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "rPPG Inference API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)