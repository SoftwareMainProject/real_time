# SW-MAIN/deeplearning/models/lightweight_cnn_v5.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    in_channels -> Global Pool -> FC(ratio) -> ReLU -> FC -> Sigmoid -> scale
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = F.adaptive_avg_pool2d(x, (1,1)).view(b, c)
        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class BottleneckSE(nn.Module):
    """
    Bottleneck + Squeeze-and-Excitation
    """
    def __init__(self, in_ch, bottleneck_ch, out_ch, stride=1, reduction=16, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, bottleneck_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_ch)
        
        self.conv2 = nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_ch)

        self.dropout = nn.Dropout2d(dropout) if dropout>0 else nn.Identity()

        self.conv3 = nn.Conv2d(bottleneck_ch, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.se = SEBlock(out_ch, reduction=reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.se(out)
        return F.relu(out)

class ImprovedLightweightCNN_v5(nn.Module):
    """
    5차 개선: Bottleneck + SEBlock
    """
    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        # 처음 Conv
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 레이어 구성
        self.layer1 = self._make_layer(64, 64, 256, stride=2, num_blocks=2)
        self.layer2 = self._make_layer(256, 64, 256, stride=2, num_blocks=2)
        self.layer3 = self._make_layer(256, 64, 256, stride=2, num_blocks=2)
        self.layer4 = self._make_layer(256, 64, 256, stride=1, num_blocks=1)

        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, feature_dim)

    def _make_layer(self, in_ch, bottleneck_ch, out_ch, stride, num_blocks):
        layers = []
        layers.append(BottleneckSE(in_ch, bottleneck_ch, out_ch, stride=stride, reduction=16, dropout=0.2))
        for _ in range(1, num_blocks):
            layers.append(BottleneckSE(out_ch, bottleneck_ch, out_ch, stride=1, reduction=16, dropout=0.2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, 112, 112]
        x = self.layer1(x)                     # [B, 256, 56, 56]
        x = self.layer2(x)                     # [B, 256, 28, 28]
        x = self.layer3(x)                     # [B, 256, 14, 14]
        x = self.layer4(x)                     # [B, 256, 14, 14]
        x = F.relu(self.bn2(self.conv2(x)))    # [B, 256, 14, 14]
        x = self.avgpool(x)                    # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)              # [B, 256]
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = ImprovedLightweightCNN_v5(in_channels=3, feature_dim=256)
    dummy_input = torch.randn(1, 3, 224, 224)
    out = model(dummy_input)
    print(out.shape)  # Expected [1, 256]
