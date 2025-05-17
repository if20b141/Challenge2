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
        # x: Tensor [1, nmels, time]
        x = x.clone()
        for _ in range(self.num_masks):
            time_mask = T.TimeMasking(time_mask_param=self.time_mask_param)
            freq_mask = T.FrequencyMasking(freq_mask_param=self.freq_mask_param)
            x = time_mask(x)
            x = freq_mask(x)
        return x