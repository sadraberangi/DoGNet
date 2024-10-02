import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureMapAttention(nn.Module):
    def __init__(self, num_feature_maps, channel_dim, attention_dim=None, middle_dim=None):
        """
        Args:
            num_feature_maps (int): Number of input feature maps (n).
            channel_dim (int): Number of channels (C) in each feature map.
            attention_dim (int, optional): Dimension for the attention mechanism. Defaults to channel_dim.
        """
        super(FeatureMapAttention, self).__init__()
        self.n = num_feature_maps
        self.C = channel_dim

        self.attention_dim = attention_dim if attention_dim is not None else channel_dim
        self.middle_dim = middle_dim if middle_dim is not None else channel_dim * 2

        self.fc1 = nn.Linear(self.C, self.middle_dim)
        self.relu = nn.ReLU()

        # Linear layers to project features to query, key, and value
        self.query = nn.Linear(self.middle_dim, self.attention_dim)
        self.key = nn.Linear(self.middle_dim, self.attention_dim)
        self.value = nn.Linear(self.middle_dim, self.attention_dim)

        # To combine the attention output back to C channels
        self.out_proj = nn.Linear(self.attention_dim, self.C)

    def forward(self, feature_maps):
        """
        Args:
            feature_maps (list or tensor): List of n feature maps, each of shape (B, C, H, W)
        
        Returns:
            Tensor: Combined feature map of shape (B, C, H, W)
        """
        # Ensure feature_maps is a tensor of shape (B, n, C, H, W)
        if isinstance(feature_maps, list):
            feature_maps = torch.stack(feature_maps, dim=1)  # Shape: (B, n, C, H, W)
        elif isinstance(feature_maps, torch.Tensor):
            # Assume already (B, n, C, H, W)
            pass
        else:
            raise TypeError("feature_maps must be a list or a tensor")

        B, n, C, H, W = feature_maps.size()
        assert n == self.n, f"Expected {self.n} feature maps, but got {n}"

        # Global Average Pooling: (B, n, C, H, W) -> (B, n, C)
        pooled = F.adaptive_avg_pool2d(feature_maps.view(B * n, C, H, W), (1,1)).view(B, n, C)

        pooled = self.relu(self.fc1(pooled))

        # Compute Query, Key, Value
        Q = self.query(pooled)  # (B, n, attention_dim)
        K = self.key(pooled)    # (B, n, attention_dim)
        V = self.value(pooled)  # (B, n, attention_dim)

        # Compute scaled dot-product attention
        # Attention scores: (B, n, n)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.attention_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)  # (B, n, n)

        # Apply attention weights to V: (B, n, attention_dim)
        attention_output = torch.matmul(attention_weights, V)  # (B, n, attention_dim)

        # Optionally, project back to C dimensions
        attention_output = self.out_proj(attention_output)  # (B, n, C)


        # Reshape weights for broadcasting: (B, n, 1, 1, 1)
        attention_output = attention_output.view(B, n, C, 1, 1)

        # Weighted sum of feature maps: sum over n
        weighted_feature_maps = feature_maps * attention_output  # (B, n, C, H, W)
        combined_feature_map = weighted_feature_maps.sum(dim=1)  # (B, C, H, W)

        return combined_feature_map

# Example Usage
if __name__ == "__main__":
    B, C, H, W = 8, 64, 32, 32
    n = 5  # Number of feature maps

    # Create dummy feature maps
    feature_maps = [torch.randn(B, C, H, W) for _ in range(n)]

    # Initialize attention module
    attention_module = FeatureMapAttention(num_feature_maps=n, channel_dim=C)

    # Forward pass
    combined_feature = attention_module(feature_maps)

    print(combined_feature.shape)  # Expected: (B, C, H, W)
