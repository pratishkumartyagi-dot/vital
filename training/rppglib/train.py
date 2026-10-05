import sys, os
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io
import matplotlib.pyplot as plt
import cv2
import torch
from sklearn.model_selection import KFold, train_test_split

import rppglib.data_utils
import rppglib.face_utils
import rppglib.params
import rppglib.processing
import rppglib.models
import rppglib.ppg2hr


class rPPG_Dataset(torch.utils.data.Dataset):
    def __init__(self, preprocessed_files, video_processing, config):
        self.config = config
        self.preprocessed_files = preprocessed_files
        self.video_processing = video_processing

    def __len__(self, ):
        return len(self.preprocessed_files)

    def __getitem__(self, index):
        path = self.preprocessed_files[index]
        data = np.load(path, allow_pickle=True)
        
        if path.endswith('.npz'):
            video = data['video']
            landmarks = data['landmarks']
            ppg = data['ppg']
        else:
            video = data
            landmarks = np.zeros((video.shape[0], 68, 2), dtype=np.int16)
            ppg = np.random.randn(video.shape[0]).astype(np.float32)

        video = self.preprocess_video(video, landmarks)
        ppg = self.preprocess_ppg(ppg)
        
        return video, ppg

    def preprocess_video(self, video, landmarks):
        video = self.video_processing(video, landmarks)
        return video

    def preprocess_ppg(self, ppg):
        ppg = rppglib.processing.bandpass_filter(ppg, self.config.fps, self.config.ppg_low_freq, self.config.ppg_high_freq)
        ppg -= ppg.mean()
        ppg /= ppg.std() + 1e-9    
        return ppg


class MIMICDataset(torch.utils.data.Dataset):
    """Dataset for MIMIC-III-Ext-PPG contact PPG signals.
    
    Loads raw PPG/ECG/ABP/RESP waveforms from WFDB format and returns
    preprocessed signal windows with regression targets (HR, RR, SBP, DBP).
    """
    
    def __init__(self, metadata_df, config, signal_types=None, target_types=None, window_seconds=30):
        self.config = config
        self.metadata = metadata_df.reset_index(drop=True)
        self.signal_types = signal_types or ['pleth', 'ecg', 'abp', 'resp']
        self.target_types = target_types or ['hr', 'rr', 'sbp', 'dbp']
        self.window_seconds = window_seconds
        self.fs = config.get('mimic_fs', 125)
        self.window_samples = int(window_seconds * self.fs)
        
        # Preprocessing parameters
        self.ppg_low_freq = config.get('ppg_low_freq', 0.5)
        self.ppg_high_freq = config.get('ppg_high_freq', 8.0)
        self.ecg_low_freq = config.get('ecg_low_freq', 0.5)
        self.ecg_high_freq = config.get('ecg_high_freq', 40.0)
        self.abp_low_freq = config.get('abp_low_freq', 0.5)
        self.abp_high_freq = config.get('abp_high_freq', 10.0)
        self.resp_low_freq = config.get('resp_low_freq', 0.1)
        self.resp_high_freq = config.get('resp_high_freq', 1.0)
        
        # Signal name mapping in WFDB files
        self.signal_name_map = {
            'pleth': ['PLETH', 'PPG', 'PLETHYSMOGRAM'],
            'ecg': ['ECG', 'ECG2', 'II', 'LEAD II'],
            'abp': ['ABP', 'ART', 'ARTERIAL'],
            'resp': ['RESP', 'RESP1', 'RESPIRATION']
        }
        
        # Cache for loaded records
        self._record_cache = {}
        self._max_cache_size = 10

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        
        # Load WFDB record
        record_path = row['folder_path'] + '/' + row['signal_file_name']
        signals, fields = self._load_record(record_path)
        
        # Extract signal channels
        signal_dict = self._extract_signals(signals, fields['sig_name'])
        
        # Preprocess each signal
        processed_signals = {}
        for sig_type in self.signal_types:
            if sig_type in signal_dict and signal_dict[sig_type] is not None:
                processed_signals[sig_type] = self._preprocess_signal(
                    signal_dict[sig_type], sig_type
                )
            else:
                # Return zeros if signal not available
                processed_signals[sig_type] = np.zeros(self.window_samples, dtype=np.float32)
        
        # Get regression targets
        targets = self._get_targets(row)
        
        # Primary input is PPG (contact)
        primary_signal = processed_signals.get('pleth', np.zeros(self.window_samples, dtype=np.float32))
        
        # Additional signals as auxiliary inputs
        aux_signals = np.stack([
            processed_signals.get('ecg', np.zeros(self.window_samples)),
            processed_signals.get('abp', np.zeros(self.window_samples)),
            processed_signals.get('resp', np.zeros(self.window_samples))
        ], axis=0)  # Shape: (3, window_samples)
        
        return {
            'ppg': torch.from_numpy(primary_signal.astype(np.float32)),
            'aux_signals': torch.from_numpy(aux_signals.astype(np.float32)),
            'targets': torch.from_numpy(targets.astype(np.float32)),
            'subject_id': row['subject_id'],
            'segment_id': row['segment_id'],
            'rhythm': row.get('event_rhythm', 'UNKNOWN')
        }

    def _load_record(self, record_path):
        """Load WFDB record with caching."""
        if record_path in self._record_cache:
            return self._record_cache[record_path]
        
        signals, fields = rppglib.data_utils.load_wfdb_record(record_path)
        
        if len(self._record_cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._record_cache))
            del self._record_cache[oldest_key]
        
        self._record_cache[record_path] = (signals, fields)
        return signals, fields

    def _extract_signals(self, signals, sig_names):
        """Extract specific signal types from multi-channel WFDB data."""
        sig_names_upper = [s.upper() for s in sig_names]
        extracted = {}
        
        for sig_type, possible_names in self.signal_name_map.items():
            found = False
            for i, name in enumerate(sig_names_upper):
                if any(pn in name for pn in possible_names):
                    extracted[sig_type] = signals[:, i].astype(np.float32)
                    found = True
                    break
            if not found:
                extracted[sig_type] = None
        
        return extracted

    def _preprocess_signal(self, signal, sig_type):
        """Apply bandpass filtering and normalization based on signal type."""
        if sig_type == 'pleth':
            low, high = self.ppg_low_freq, self.ppg_high_freq
        elif sig_type == 'ecg':
            low, high = self.ecg_low_freq, self.ecg_high_freq
        elif sig_type == 'abp':
            low, high = self.abp_low_freq, self.abp_high_freq
        elif sig_type == 'resp':
            low, high = self.resp_low_freq, self.resp_high_freq
        else:
            low, high = 0.5, 8.0
        
        # Bandpass filter
        signal = rppglib.processing.bandpass_filter(signal, self.fs, low, high, order=4)
        
        # Normalize
        signal = signal - np.mean(signal)
        signal = signal / (np.std(signal) + 1e-9)
        
        # Ensure correct length
        if len(signal) > self.window_samples:
            signal = signal[:self.window_samples]
        elif len(signal) < self.window_samples:
            signal = np.pad(signal, (0, self.window_samples - len(signal)), mode='constant')
        
        return signal.astype(np.float32)

    def _get_targets(self, row):
        """Extract regression targets from metadata row."""
        targets = []
        
        if 'hr' in self.target_types:
            hr = row.get('median_30s_hr', np.nan)
            targets.append(hr if not np.isnan(hr) else 0.0)
        
        if 'rr' in self.target_types:
            rr = row.get('median_30s_rr', np.nan)
            targets.append(rr if not np.isnan(rr) else 0.0)
        
        if 'sbp' in self.target_types:
            sbp = row.get('median_30s_sbp', np.nan)
            targets.append(sbp if not np.isnan(sbp) else 0.0)
        
        if 'dbp' in self.target_types:
            dbp = row.get('median_30s_dbp', np.nan)
            targets.append(dbp if not np.isnan(dbp) else 0.0)
        
        return np.array(targets, dtype=np.float32)


def collate_mimic_batch(batch):
    """Custom collate function for MIMIC dataset."""
    ppg = torch.stack([item['ppg'] for item in batch])
    aux_signals = torch.stack([item['aux_signals'] for item in batch])
    targets = torch.stack([item['targets'] for item in batch])
    
    # Keep metadata as lists
    subject_ids = [item['subject_id'] for item in batch]
    segment_ids = [item['segment_id'] for item in batch]
    rhythms = [item['rhythm'] for item in batch]
    
    return {
        'ppg': ppg,
        'aux_signals': aux_signals,
        'targets': targets,
        'subject_ids': subject_ids,
        'segment_ids': segment_ids,
        'rhythms': rhythms
    }

def train_fold(config):
    model = rppglib.models.__dict__[config.model](config)
    print(f'Model {config.model} loaded')
    
    if model.require_train:
        
        print(f'Training on fold {config.valid_fold}')
        df = pd.read_csv(f'{config.train_dataset}.csv')

        test_mask = df['fold'] == config.test_fold
        valid_mask = df['fold'] == config.valid_fold
        train_mask = (~test_mask)&(~valid_mask)
    
        print('Train files:', train_mask.sum())
        print('Valid files:', valid_mask.sum())
        
        train_ds = rPPG_Dataset(df.loc[train_mask, 'file'].values, model.video_processing, config)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
        
        valid_ds = rPPG_Dataset(df.loc[valid_mask, 'file'].values, model.video_processing, config)
        valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
        model.train(train_dl, valid_dl)

    config.test_results = dict()
    for test_dataset in config.test_datasets:
        print()
        print(f'Testing on {test_dataset}')
        df = pd.read_csv(f'{test_dataset}.csv')
        test_mask = df['fold'] == config.test_fold
        print('Test files:', test_mask.sum())
        test_ds = rPPG_Dataset(df.loc[test_mask, 'file'].values, model.video_processing, config)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)


        true_ppgs_epoch = list()
        pred_ppgs_epoch = list()

        for videos, true_ppgs in tqdm(test_dl, desc='Testing', file=sys.stdout):
            assert not torch.isnan(videos).any(), 'NaNs in Videos'
            assert not torch.isnan(true_ppgs).any(), 'NaNs in PPGs'
            pred_ppgs = model.predict(videos)
            assert not torch.isnan(pred_ppgs).any(), 'NaNs in pred PPGs'
            
            true_ppgs_epoch.append(true_ppgs.detach().cpu().numpy())
            pred_ppgs_epoch.append(pred_ppgs.detach().cpu().numpy())

        
        true_ppgs_epoch = np.concatenate(true_ppgs_epoch, axis=0)
        pred_ppgs_epoch = np.concatenate(pred_ppgs_epoch, axis=0)


        true_ppgs_epoch = true_ppgs_epoch[:, :pred_ppgs_epoch.shape[1]]
        print('Testing true PPGs:', true_ppgs_epoch.shape)
        print('Testing pred PPGs:', pred_ppgs_epoch.shape)

        ppg_mae, hr_mae = calc_metrics(true_ppgs_epoch, pred_ppgs_epoch)
        
        config.test_results[f'test__{test_dataset}__ppg'] = ppg_mae
        config.test_results[f'test__{test_dataset}__hr'] = hr_mae
        print('PPG MAE:', ppg_mae)
        print('HR MAE:', hr_mae)
    return config

def calc_metrics(true_ppgs, pred_ppgs):

    ppg_mae = np.abs(pred_ppgs - true_ppgs).mean(axis=1).mean().item()

    true_hrs = list()
    pred_hrs = list()
    for i in range(true_ppgs.shape[0]):
        true = true_ppgs[i]
        pred = pred_ppgs[i]

        true_hr = rppglib.processing.calculate_fft_hr(true)
        pred_hr = rppglib.processing.calculate_fft_hr(pred)

        true_hrs.append(true_hr)
        pred_hrs.append(pred_hr)

    print('Mean true HRs:', np.mean(true_hrs))
    print('Mean pred HRs:', np.mean(pred_hrs))
        
        
    hr_mae = np.abs(np.array(true_hrs) - np.array(pred_hrs)).mean()
    return ppg_mae, hr_mae
