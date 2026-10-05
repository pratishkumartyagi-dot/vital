import skvideo.io
import numpy as np
import pandas as pd
from pathlib import Path

import numpy
numpy.float = numpy.float64
numpy.int = numpy.int_

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    wfdb = None

def load_video(path):
    videodata = skvideo.io.vread(path) 
    return videodata


def load_ppg(path):
    with open(path, 'rt') as f:
        ppg = f.read().split('\n')[:-1]
    ppg = np.array([int(line.split()[0]) for line in ppg]).astype('float32')
    return ppg


def load_wfdb_record(record_path):
    """Load a WFDB record (header + data files).
    
    Args:
        record_path: Path to the record without extension (e.g., 'path/to/record')
    
    Returns:
        tuple: (signals, fields) where signals is (n_samples, n_channels) array
               and fields is a dict with metadata including fs, sig_name, units, etc.
    """
    if not WFDB_AVAILABLE:
        raise ImportError("wfdb package required for MIMIC-III-Ext-PPG. Install with: pip install wfdb")
    
    record = wfdb.rdrecord(record_path)
    return record.p_signal, {
        'fs': record.fs,
        'sig_name': record.sig_name,
        'units': record.units,
        'n_sig': record.n_sig,
        'sig_len': record.sig_len,
        'comments': record.comments
    }


def load_mimic_metadata(metadata_csv_path):
    """Load MIMIC-III-Ext-PPG metadata CSV.
    
    Args:
        metadata_csv_path: Path to metadata.csv file
    
    Returns:
        pd.DataFrame with all segment metadata
    """
    return pd.read_csv(metadata_csv_path)


def filter_mimic_segments(df, sqi_threshold=1, signal_types=None):
    """Filter MIMIC segments by signal quality.
    
    Args:
        df: Metadata DataFrame
        sqi_threshold: Minimum SQI value (1=high quality, 0=usable, negative=poor)
        signal_types: List of signal types to check SQI for (e.g., ['pleth', 'ecg', 'abp', 'resp'])
    
    Returns:
        Filtered DataFrame
    """
    if signal_types is None:
        signal_types = ['pleth']
    
    mask = pd.Series(True, index=df.index)
    
    for sig in signal_types:
        sqi_col = f'vector_10s_{sig}_sqi' if sig != 'resp' else 'resp_sqi'
        if sqi_col in df.columns:
            if sig == 'resp':
                mask &= (df[sqi_col] >= sqi_threshold)
            else:
                mask &= (df[sqi_col].apply(lambda x: np.min(x) >= sqi_threshold if isinstance(x, (list, np.ndarray)) else x >= sqi_threshold))
    
    return df[mask].reset_index(drop=True)


def get_mimic_segment_paths(df, base_path):
    """Get full paths to WFDB files from metadata DataFrame.
    
    Args:
        df: Metadata DataFrame
        base_path: Base path to MIMIC-III-Ext-PPG data directory
    
    Returns:
        List of full paths to .hea files (without extension)
    """
    base_path = Path(base_path)
    paths = []
    for _, row in df.iterrows():
        folder = base_path / row['folder_path']
        signal_file = row['signal_file_name']
        full_path = folder / signal_file
        paths.append(str(full_path))
    return paths


def parse_mimic_sqi_vector(sqi_val):
    """Parse SQI vector from metadata (can be string representation of array).
    
    Args:
        sqi_val: SQI value from metadata (string, list, or array)
    
    Returns:
        numpy array of SQI values
    """
    if isinstance(sqi_val, str):
        sqi_val = eval(sqi_val)
    return np.array(sqi_val)
