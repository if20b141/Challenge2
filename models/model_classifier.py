import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioMLP(nn.Module):
    def __init__(self, n_mels, output_size, time_reduce=1):
        super().__init__()
        self.time_reduce = time_reduce
        self.pool = nn.AvgPool1d(kernel_size=time_reduce, stride=time_reduce) if time_reduce > 1 else nn.Identity()

        # Convolutional feature extractor
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # (B, 1, n_mels, time)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))  # Halbiert Frequenz und Zeit
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        # Adaptive pooling for arbitrary input length
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected classification head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        # x shape: (B, 1, n_mels, time)
        if self.time_reduce > 1:
            # reduce time dimension
            b, c, h, w = x.shape
            x = x.view(b * c * h, 1, w)
            x = self.pool(x)
            new_w = x.shape[-1]
            x = x.view(b, c, h, new_w)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x
