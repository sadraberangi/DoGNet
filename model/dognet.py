import torch
import torch.nn as nn
from dog import DoG
from unfis import UNFIS
from feature_aggregator import FeatureMapAttention
import torch.nn.functional as F


class DoGNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, num_pyramid_levels=3, drop_out=0.5):  # Updated num_classes
        super(DoGNet, self).__init__()
        self.num_pyramid_levels = num_pyramid_levels

        self.channels = [16, 32, 32, 64, 64, 128, 128, 256, 256]

        # First custom convolutional layer  
        self.dogs = nn.ModuleList([
            DoG(in_channels=in_channels, out_channels=self.channels[0],
                kernel_size=5, stride=1, padding=2)
            for _ in range(num_pyramid_levels)
        ])

        self.atten = FeatureMapAttention(num_pyramid_levels, self.channels[0])

        self.conv2 = nn.Conv2d(
            in_channels=self.channels[0], out_channels=self.channels[1], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=self.channels[1], out_channels=self.channels[2], kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(self.channels[2])

        self.conv4 = nn.Conv2d(
            in_channels=self.channels[2], out_channels=self.channels[3], kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=self.channels[3], out_channels=self.channels[4], kernel_size=5, stride=1, padding=2)
        self.conv6 = nn.Conv2d(
            in_channels=self.channels[4], out_channels=self.channels[5], kernel_size=3, stride=1, padding=1)

        self.bn2 = nn.BatchNorm2d(self.channels[5])

        self.conv7 = nn.Conv2d(
            in_channels=self.channels[5], out_channels=self.channels[6], kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(
            in_channels=self.channels[6], out_channels=self.channels[7], kernel_size=5, stride=1, padding=2)
        self.conv9 = nn.Conv2d(
            in_channels=self.channels[7], out_channels=self.channels[8], kernel_size=3, stride=1, padding=1)

        self.bn3 = nn.BatchNorm2d(self.channels[8])

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(self.channels[8], num_classes)  # Updated num_classes

        self.unfis = UNFIS(self.channels[8], 5, num_classes) #unfis instead of fc1

        self.max_pooling2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        
        pyramid = self.generate_pyramid(x, self.num_pyramid_levels)

        pyramid_features = []
        for i, level in enumerate(pyramid):
            features = self.dogs[i](level)
            features = F.relu(features)

            if i > 0:
                features = F.interpolate(
                    features,
                    size=pyramid[0].shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            pyramid_features.append(features)

        pyramid_features = torch.stack(pyramid_features, dim=1)

        combined_features = self.atten(pyramid_features)

        x = self.relu(self.conv2(combined_features))
        x = self.relu(self.bn1(self.conv3(x)))

        x = self.max_pooling2d(x)

        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.bn2(self.conv6(x)))

        x = self.max_pooling2d(x)

        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.bn3(self.conv9(x)))

        x = self.global_avg_pool(x).view(-1, self.channels[-1])
        # x = self.fc1(x)
        x = self.unfis(x) # instead of fc1

        return x

    def generate_pyramid(self, image, num_levels):
        pyramid = [image]   
        for i in range(1, num_levels):
            downsampled_image = F.interpolate(
                pyramid[-1], scale_factor=0.5, mode='bilinear', align_corners=False)
            pyramid.append(downsampled_image)
        return pyramid


# Example Usage
if __name__ == "__main__":
    # Create a sample input tensor
    batch_size = 2
    in_channels = 3
    height, width = 64, 64
    x = torch.randn(batch_size, in_channels, height, width).to('cpu')

    model = DoGNet(in_channels=in_channels).to('cpu')

    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")  # Expected: (2, 16, 64, 64)
