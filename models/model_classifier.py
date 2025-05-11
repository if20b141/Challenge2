import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioMLP(nn.Module):
    def __init__(self, n_steps, n_mels, hidden1_size, hidden2_size, output_size, time_reduce=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_reduce = time_reduce
        # optimized for GPU, faster than x.reshape(*x.shape[:-1], -1, 2).mean(-1)
        self.pool = nn.AvgPool1d(kernel_size=time_reduce, stride=time_reduce)  # Non-overlapping averaging

        self.fc1 = nn.Linear(n_steps * n_mels, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # reduce time dimension
        shape = x.shape
        x = x.reshape(-1, 1, x.shape[-1])
        x = self.pool(x)  # (4096, 1, 431//n)
        x = x.reshape(shape[0], shape[1], shape[2], -1)

        # 2D to 1D
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ESC50CNN(nn.Module):
    def __init__(self, n_mels=128, n_steps=431, output_size=50):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # -> (B, 32, 128, 431)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # -> (B, 32, 64, 215)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> (B, 64, 64, 215)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # -> (B, 64, 32, 107)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # -> (B, 128, 32, 107)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # -> (B, 128, 16, 53)
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 53, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        # x: (B, 1, 128, 431)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=50, dropout=0.3):
        super().__init__()
        self.in_channels = 64

        # ⬇️ Verbesserte conv1 für Spectrogramme
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Input: (B, 1, 128, ~431)
        x = self.conv1(x)       # -> (B, 64, 128, ~431)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # -> (B, 64, 64, ~216)

        x = self.layer1(x)      # -> (B, 256, 64, ~216)
        x = self.layer2(x)      # -> (B, 512, 32, ~108)
        x = self.layer3(x)      # -> (B, 1024, 16, ~54)
        x = self.layer4(x)      # -> (B, 2048, 8, ~27)

        x = self.avgpool(x)     # -> (B, 2048, 1, 1)
        x = torch.flatten(x, 1) # -> (B, 2048)
        x = self.dropout(x)
        x = self.fc(x)          # -> (B, 50)
        return x
