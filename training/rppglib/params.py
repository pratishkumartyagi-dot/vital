import argparse

class Config:
    def __init__(self):
        self.fps = 30
        self.ppg_low_freq = 0.5
        self.ppg_high_freq = 3.5
        self.batch_size = 36
        self.num_workers = 0
        self.train_dataset = 'mcd_rppg'
        self.test_datasets = ['mcd_rppg']
        self.num_folds = 5
        self.test_fold = 0
        self.valid_fold = 1
        self.model = 'SCNN_8rois'
        self.results_folder = 'results'
        self.device = 'cpu'

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SCNN_8rois')
    parser.add_argument('--train_dataset', type=str, default='mcd_rppg')
    parser.add_argument('--test_datasets', type=str, nargs='+', default=['mcd_rppg'])
    parser.add_argument('--valid_fold', type=int, default=1)
    parser.add_argument('--test_fold', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--results_folder', type=str, default='results')
    
    args = parser.parse_args()
    config = Config()
    for key, value in vars(args).items():
        setattr(config, key, value)
    return config
