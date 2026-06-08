import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),            
        )
    
    def forward(self, x):
        return self.block(x)


class QuickDrawCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1,32),
            nn.MaxPool2d(2),
            
            ConvBlock(32,64),
            nn.MaxPool2d(2),

        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(
                in_features=128,
                out_features=num_classes
            )
        )
    

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x