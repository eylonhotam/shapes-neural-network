import torch.nn as nn


class ShapeWaveNet(nn.Module):
    """
    Lightweight CNN for classifying hand-drawn shapes (Square, Triangle, Circle).
    Trained on synthetic 64x64 grayscale images generated with OpenCV.

    Architecture:
        Input [1x64x64]
        -> Conv2d(1->16) + ReLU + MaxPool  -> [16x32x32]
        -> Conv2d(16->32) + ReLU + MaxPool -> [32x16x16]
        -> Dropout(0.25)
        -> Flatten -> Linear(8192->128) + ReLU
        -> Dropout(0.50)
        -> Linear(128->3)
        -> [Square | Triangle | Circle]
    """

    def __init__(self):
        super(ShapeWaveNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

