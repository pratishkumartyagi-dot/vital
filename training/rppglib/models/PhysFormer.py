import torch
import torch.nn as nn
import numpy as np
from .base_model import BaseModel

class PhysFormerModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

class PhysFormer(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.net = PhysFormerModel()
        self.weights_name = f'{config.results_folder}/{config.model}__{config.train_dataset}__{config.valid_fold}.pt'
        
    def predict(self, videos):
        return videos.mean(dim=1)
    
    def video_processing(self, video, landmarks):
        return np.mean(video, axis=(1,2))
