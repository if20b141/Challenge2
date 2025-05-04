import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioMLP(nn.Module):
    def __init__(self, n_steps, n_mels, hidden1_size, hidden2_size, output_size, time_reduce=1):
        super().__init__()
        self.time_reduce = time_reduce
        self.n_steps = n_steps
        self.n_mels = n_mels

        self.pool = nn.AvgPool1d(kernel_size=time_reduce, stride=time_reduce) if time_reduce > 1 else nn.Identity()

        # Feature extractor
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (B, 64, 1, 1)
        )

        # Fully connected head
        self.fc = nn.Sequential(
            nn.Flatten(),  # -> (B, 64)
            nn.Linear(64, hidden1_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden1_size, hidden2_size),
            nn.ReLU(),
            nn.Linear(hidden2_size, output_size)
        )

    def forward(self, x):
        # x shape: (B, n_mels, n_steps)
        b = x.shape[0]
        x = x.view(b, 1, self.n_mels, self.n_steps)  # (B, 1, n_mels, time)

        if self.time_reduce > 1:
            x = x.view(b, 1 * self.n_mels, self.n_steps)  # (B, C=1*n_mels, time)
            x = self.pool(x)
            new_time = x.shape[-1]
            x = x.view(b, 1, self.n_mels, new_time)

        x = self.conv_block(x)
        x = self.fc(x)
        return x
