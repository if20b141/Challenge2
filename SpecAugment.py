import torch
import torchaudio.transforms as T
import random

class SpecAugment(torch.nn.Module):
    def __init__(self, time_mask_param=30, freq_mask_param=13, num_masks=2):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks

    def forward(self, x):
        x = x.clone()
        for _ in range(self.num_masks):
            x = T.TimeMasking(time_mask_param=self.time_mask_param)(x)
            x = T.FrequencyMasking(freq_mask_param=self.freq_mask_param)(x)
        return x