import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.require_train = True
        
    def forward(self, x):
        raise NotImplementedError
        
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def video_processing(self, video, landmarks):
        raise NotImplementedError
