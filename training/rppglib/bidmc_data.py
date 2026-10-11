"""
BIDMC (Beth Israel Deaconess Medical Centre) contact-PPG dataset loader
for ROSE-Net cross-domain pre-training.

The BIDMC dataset contains 8-minute recordings (53 subjects, fs = 125 Hz)
of ECG, PPG (pleth), and impedance respiration signals, along with
annotation parameters: HR, RR, SpO2 (sampled at 1 Hz).

To bridge the domain gap between contact PPG and camera-based rPPG,
we:
  1. Load 125 Hz PPG waveforms and 1 Hz SpO2 reference values.
  2. Resample PPG to 30 Hz (camera frame rate).
  3. Synthesise RGB signals from PPG + SpO2 using Beer-Lambert modelling
     (see ``rppglib.stmap.synthesize_rgb_from_ppg``).
  4. Generate STMaps from the synthetic RGB signals.
  5. Yield (stmap, spo2) pairs for supervised training.
"""
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from scipy.signal import resample, butter, filtfilt

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False


class BIDMCDataset:
    """Loader for BIDMC PPG + SpO2 reference data.

    Supports both PhysioBank/CinC WFDB format and the bundled
    ``bidmc_data.mat`` (Matlab format).

    Args:
        data_path: Path to the BIDMC dataset root directory.
        fs_target: Target sampling rate (camera frame rate, default 30 Hz).
        window_sec: Duration of each training window.
        overlap:    Fractional overlap between consecutive windows.
    """

    DEFAULT_SIGNAL_NAMES = ["pleth", "v", "ii", "resp"]

    def __init__(
        self,
        data_path: str,
        fs_target: float = 30.0,
        window_sec: float = 8.0,
        overlap: float = 0.5,
    ):
        self.data_path = Path(data_path)
        self.fs_target = fs_target
        self.window_sec = window_sec
        self.overlap = overlap
        self.window_samples = int(window_sec * fs_target)

        self.records: List[Tuple[np.ndarray, float]] = []
        self._load_all()

    def _load_all(self):
        """Load all BIDMC records and pre-segment into windows."""
        # Try WFDB format first
        hea_files = sorted(self.data_path.glob("*.hea"))
        if hea_files and WFDB_AVAILABLE:
            for hea in hea_files:
                self._load_wfdb_record(hea.stem)
        else:
            # Try numpy cache or mat files
            npz_files = sorted(self.data_path.glob("*.npz"))
            for npz in npz_files:
                self._load_npz_record(npz)
            mat_files = sorted(self.data_path.glob("bidmc_data.mat"))
            for mat in mat_files:
                self._load_mat_record(mat)

        if not self.records:
            print("Warning: No BIDMC records found. Training will use synthetic data only.")

    def _load_wfdb_record(self, record_stem: str):
        """Load a single BIDMC record in WFDB format."""
        try:
            record_path = str(self.data_path / record_stem)
            record = wfdb.rdrecord(record_path)
            fields = record.__dict__

            fs = record.fs
            signals = record.p_signal

            # Determine which channel is PPG
            sig_names = [s.upper() if s else f"ch{i}" for i, s in enumerate(fields.get("sig_name", []))]
            ppg_idx = self._find_signal_channel(sig_names, ["PLETH", "PPG"])
            spo2_idx = self._find_signal_channel(sig_names, ["SPO2", "SAO2"])

            if ppg_idx is None or spo2_idx is None:
                return

            ppg = signals[:, ppg_idx]
            spo2_vals = signals[:, spo2_idx]

            # Get sampling rate
            fs_record = fs if fs else 125.0

            self._segment_and_store(ppg, spo2_vals, fs_record)
        except Exception as e:
            print(f"  Warning: Failed to load {record_stem}: {e}")

    def _load_npz_record(self, npz_path: Path):
        """Load a preprocessed .npz record with 'ppg' and 'spo2' arrays."""
        try:
            data = np.load(npz_path, allow_pickle=True)
            ppg = data["ppg"]
            spo2 = data["spo2"] if "spo2" in data.files else data["ref_spo2"]
            fs = float(data.get("fs", 125.0))
            self._segment_and_store(ppg, spo2, fs)
        except Exception as e:
            print(f"  Warning: Failed to load {npz_path}: {e}")

    def _load_mat_record(self, mat_path: Path):
        """Load the bundled bidmc_data.mat file."""
        try:
            import scipy.io as sio
            mat = sio.loadmat(mat_path)
            data = mat["data"]

            for entry in data[0]:
                ppg_struct = entry["ppg"]
                ppg = ppg_struct["v"].squeeze()
                fs = float(ppg_struct["fs"].squeeze())

                ref = entry["ref"]
                spo2 = ref["params"]["spo2"].squeeze()

                self._segment_and_store(ppg, spo2, fs)
        except Exception as e:
            print(f"  Warning: Failed to load {mat_path}: {e}")

    @staticmethod
    def _find_signal_channel(sig_names: List[str], targets: List[str]):
        for i, name in enumerate(sig_names):
            for target in targets:
                if target in name:
                    return i
        return None

    def _segment_and_store(self, ppg: np.ndarray, spo2_ref: np.ndarray, fs: float):
        """Resample PPG to target rate, segment into windows, and store."""
        from ..stmap import synthesize_rgb_from_ppg, generate_stmap

        # Resample PPG to target sampling rate
        n_target = int(len(ppg) * self.fs_target / fs)
        ppg_rs = resample(ppg, n_target)
        fs = self.fs_target

        # SpO2 reference is at 1 Hz; interpolate to match PPG length
        if len(spo2_ref) > 1:
            t_spo2 = np.linspace(0, 1, len(spo2_ref))
            t_ppg = np.linspace(0, 1, len(ppg_rs))
            spo2_interp = np.interp(t_ppg, t_spo2, spo2_ref)
        else:
            spo2_interp = np.full(len(ppg_rs), spo2_ref[0] if len(spo2_ref) > 0 else 95.0)

        # Estimate HR per window for AC/DC synthesis
        from scipy.signal import find_peaks
        ppg_filt = filtfilt(
            *butter(2, [0.5 / (fs / 2), 3.5 / (fs / 2)], btype="band"),
            ppg_rs.astype(np.float64),
        ) if len(ppg_rs) > 1 else ppg_rs

        # Segment into windows
        step = int(self.window_samples * (1 - self.overlap))
        for start in range(0, len(ppg_rs) - self.window_samples, step):
            end = start + self.window_samples
            ppg_win = ppg_rs[start:end]
            spo2_win = np.median(spo2_interp[start:end])

            # Skip invalid SpO2 values
            if spo2_win < 50 or spo2_win > 100:
                continue

            # Estimate HR in this window
            peaks, _ = find_peaks(ppg_filt[start:end], distance=int(fs * 0.4))
            if len(peaks) >= 2:
                hr_est = 60.0 / np.mean(np.diff(peaks) / fs)
            else:
                hr_est = 72.0

            # Generate STMap
            rgb = synthesize_rgb_from_ppg(ppg_win, spo2_win, fs, hr_est)
            stmap = generate_stmap(rgb, target_length=self.window_samples)
            self.records.append((stmap, float(spo2_win)))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, float]:
        return self.records[idx]

    def get_train_test_split(self, test_frac: float = 0.2, seed: int = 42):
        """Return train/test indices with subject-wise split if metadata available."""
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(self.records))
        n_test = int(len(indices) * test_frac)
        return indices[n_test:], indices[:n_test]


def load_bidmc_metadata(metadata_csv: str) -> np.ndarray:
    """Load BIDMC metadata CSV with SpO2 reference values.

    Expected columns: ['file', 'spo2', 'sbp', 'dbp', 'hr', 'subject']
    """
    import pandas as pd
    df = pd.read_csv(metadata_csv)
    return df
