import numpy as np
from . import processing

class POS:
    def __init__(self, fs=30):
        self.fs = fs
        
    def run(self, rgb_signals):
        mean_rgb = np.mean(rgb_signals, axis=0)
        normalized = rgb_signals / (mean_rgb + 1e-6)
        X = normalized.T
        H = np.array([[0, 1, -1], [-2, 1, 1], [-1, 2, -1]])
        S = np.dot(H, X.T)
        std_S = np.std(S[0, :]) / (np.std(S[1, :]) + 1e-6)
        S = S[0, :] + std_S * S[1, :]
        S = processing.bandpass_filter(S, self.fs, 0.5, 3.5)
        return S

class ICA:
    def __init__(self):
        pass
        
    def run(self, rgb_signals):
        return np.mean(rgb_signals, axis=0)

class Green:
    def __init__(self):
        pass
        
    def run(self, rgb_signals):
        return rgb_signals[1, :]
