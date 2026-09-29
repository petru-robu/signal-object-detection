"""ResNet built from scratch: squeeze-excite block, residual block, network."""

import torch
import torch.nn.functional as F
from torch import nn


class SEBlock(nn.Module):
    """Squeeze-and-excitation: rescale each channel by a learned weight in (0, 1)."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class ResidualBlock(nn.Module):
    """out = act(F(x) + shortcut(x)), where F is conv-bn-act-conv-bn-SE."""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # 1x1 projection when the channel count changes, plain identity otherwise
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(self.se(out))
        return F.silu(out + self.shortcut(x))


def make_stage(in_channels, out_channels, dropout):
    return nn.Sequential(
        ResidualBlock(in_channels, out_channels, dropout),
        ResidualBlock(out_channels, out_channels, dropout),
    )


class ResNet(nn.Module):
    """Input (N, 1, 128, 55), output (N, num_classes) logits."""

    def __init__(self, in_channels=1, num_classes=5, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            # stem, 128 x 55
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            make_stage(32, 32, 0.05),
            nn.MaxPool2d((1, 2)),  # 128 x 27
            make_stage(32, 64, 0.05),
            nn.MaxPool2d((1, 2)),  # 128 x 13
            make_stage(64, 128, 0.10),
            nn.MaxPool2d(2),  # 64 x 6
            make_stage(128, 192, 0.10),
            nn.MaxPool2d(2),  # 32 x 3
            make_stage(192, 256, 0.15),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        return self.classifier(x)
