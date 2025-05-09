import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioMLP(nn.Module):
    def __init__(self, n_steps, n_mels, hidden1_size, hidden2_size, output_size, time_reduce=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_reduce = time_reduce

        # Zeitreduktion mit AvgPool2d über die Zeitachse
        self.pool = nn.AvgPool2d(kernel_size=(1, time_reduce), stride=(1, time_reduce))

        # CNN für Feature Extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input: (B, 1, n_mels, time)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                # → (B, 32, n_mels//2, time//2)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                # → (B, 64, n_mels//4, time//4)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                # → (B, 128, n_mels//8, time//8)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                # → (B, 256, 1, 1)
        )

        # MLP-Architektur für Klassifikation
        self.fc1 = nn.Linear(256, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        self.dropout = nn.Dropout(0.5)  # Erhöhte Dropout-Rate

    def forward(self, x):
        # x erwartet (B, n_mels, n_steps) oder (B, 1, n_mels, n_steps)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # → (B, 1, n_mels, n_steps)
        elif x.dim() != 4:
            raise ValueError(f"Expected input of shape (B, n_mels, n_steps) or (B, 1, n_mels, n_steps), but got {x.shape}")

        x = self.pool(x)      # Zeitreduktion → (B, 1, n_mels, reduced_steps)
        x = self.cnn(x)       # CNN → (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten → (B, 256)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout anwenden
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
