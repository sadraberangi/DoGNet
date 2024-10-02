import torch 
import torch.nn as nn 
from .dog import DoG



class SimpleCNN(nn.Module):
    def __init__(self, num_pyramid_levels=3, num_classes=9):  # Updated num_classes
        super(SimpleCNN, self).__init__()
        self.num_pyramid_levels = num_pyramid_levels

        # First custom convolutional layer
        self.convs = nn.ModuleList([
            DoG(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
            for _ in range(num_pyramid_levels)
        ])

        self.conv2 = nn.Conv2d(in_channels=32 * num_pyramid_levels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(32, num_classes)  # Updated num_classes
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        pyramid = generate_pyramid(x, self.num_pyramid_levels)

        pyramid_features = []
        for i, level in enumerate(pyramid):
            features = self.convs[i](level)
            features = F.relu(features)

            if i > 0:
                features = F.interpolate(
                    features,
                    size=pyramid[0].shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            pyramid_features.append(features)

        combined_features = torch.cat(pyramid_features, dim=1)
        
        x = self.conv2(combined_features)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.global_max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x