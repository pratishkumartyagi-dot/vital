import numpy as np
from scipy import signal

class BVPsignal:
    def __init__(self, data, fps=30, bpFilter=False, window=20):
        self.data = data
        self.fps = fps
        self.bpFilter = bpFilter
        self.window = window
        
    def getBPM(self, winsize=20):
        if len(self.data) < self.fps * self.window:
            return np.array([0.0])
        
        # Simple FFT-based HR estimation
        fft_data = np.fft.rfft(self.data)
        freqs = np.fft.rfftfreq(len(self.data), 1/self.fps)
        mask = (freqs >= 0.5) & (freqs <= 3.5)
        if not np.any(mask):
            return np.array([0.0])
        peak_idx = np.argmax(np.abs(fft_data[mask]))
        hr = freqs[mask][peak_idx] * 60
        return np.array([hr])
