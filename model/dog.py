import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DoG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, device=None, dtype=None):
        """
        Difference of Gaussian (DoG) module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the Gaussian kernel (must be odd).
            stride (int, optional): Stride for the convolution. Default is 1.
            padding (int, optional): Zero-padding added to both sides of the input. Default is 0.
        """
        super(DoG, self).__init__()

        self.factory_kwargs = factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize sigma1 and sigma2 as learnable parameters
        # Initialized such that sigma2 > sigma1
        initial_sigma1 = 1.0
        initial_sigma2 = 1.6
        self.sigma1 = nn.Parameter(torch.ones(out_channels) * initial_sigma1)
        self.sigma2 = nn.Parameter(torch.ones(out_channels) * initial_sigma2)

        self._build_grid()

    def _build_grid(self):

        # Create a grid for the Gaussian kernel
        k = self.kernel_size
        half_k = k // 2
        # Create coordinate grid
        x_coords = torch.arange(-half_k, half_k + 1, **self.factory_kwargs)
        y_coords = torch.arange(-half_k, half_k + 1, **self.factory_kwargs)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords)
        # Shape: (k, k)

        # Reshape for broadcasting
        self.x_grid = x_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
        self.y_grid = y_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)

    def forward(self, x):
        """
        Forward pass for the DoG module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying DoG convolution.
        """
        if self.x_grid.device != x.device:
          self.x_grid = self.x_grid.to(x.device)
          self.y_grid = self.y_grid.to(x.device)

        # Ensure sigma2 > sigma1 to create a proper DoG
        sigma1 = torch.clamp(self.sigma1, min=1e-5)
        sigma2 = torch.clamp(self.sigma2, min=1e-5)
        sigma_diff = sigma2 - sigma1
        sigma_diff = torch.clamp(sigma_diff, min=1e-5)
        sigma2 = sigma1 + sigma_diff  # Ensure sigma2 > sigma1

        # Compute Gaussian kernels for sigma1 and sigma2
        # Shape for sigma: (out_channels, 1, 1, 1)
        sigma1 = sigma1.view(self.out_channels, 1, 1, 1)
        sigma2 = sigma2.view(self.out_channels, 1, 1, 1)

        # Gaussian with sigma1
        exponent1 = -(self.x_grid ** 2 + self.y_grid ** 2) / (2 * sigma1 ** 2)
        g1 = torch.exp(exponent1) / (2 * math.pi * sigma1 ** 2)

        # Gaussian with sigma2
        exponent2 = -(self.x_grid ** 2 + self.y_grid ** 2) / (2 * sigma2 ** 2)
        g2 = torch.exp(exponent2) / (2 * math.pi * sigma2 ** 2)

        # Difference of Gaussian
        dog = g1 - g2  # (out_channels, 1, k, k)

        # Normalize the DoG kernels to have zero mean
        dog = dog - dog.mean(dim=[2,3], keepdim=True)

        # Expand DoG kernels to match in_channels
        # Each output channel has its own DoG filter applied to each input channel
        # Shape: (out_channels, in_channels, k, k)
        dog = dog.repeat(1, self.in_channels, 1, 1)

        # Optionally, you can initialize the convolution bias or other parameters here
        # For simplicity, we are not using bias in this implementation

        # Perform convolution
        out = F.conv2d(x, dog, bias=None, stride=self.stride, padding=self.padding)

        return out

# Example Usage
if __name__ == "__main__":
    # Create a sample input tensor
    batch_size = 2
    in_channels = 3
    height, width = 64, 64
    x = torch.randn(batch_size, in_channels, height, width)

    # Define DoG module
    out_channels = 16
    kernel_size = 7  # Must be odd
    dog = DoG(in_channels, out_channels, kernel_size, stride=1, padding=3)

    # Forward pass
    output = dog(x)
    print(f"Output shape: {output.shape}")  # Expected: (2, 16, 64, 64)
