from __future__ import annotations

import torch
import torch.nn as nn


class Model2CNN(nn.Module):
    """
    Part 1 - Second model (library-based CNN).

    Architecture constraints implemented:
    - Convolution layers with valid convolution (padding=0), ReLU, and 2x2 max pooling.
    - Flattening operation after convolution blocks.
    - One hidden fully connected layer with Sigmoid.
    - Output layer with Softmax.
    """

    def __init__(
        self,
        num_classes: int,
        input_size: int,
        kernel_sizes: list[int],
        channels: list[int],
        hidden_units: int,
    ) -> None:
        super().__init__()

        if len(kernel_sizes) != len(channels):
            raise ValueError("kernel_sizes and channels must have the same length")

        blocks = []
        in_channels = 3
        for k, out_channels in zip(kernel_sizes, channels):
            blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=1, padding=0))
            blocks.append(nn.ReLU())
            blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()

        # Compute flatten dimension once using a dummy input tensor.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            conv_out = self.feature_extractor(dummy)
            flatten_dim = int(conv_out.numel())

        self.fc_hidden = nn.Linear(flatten_dim, hidden_units)
        self.fc_out = nn.Linear(hidden_units, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = torch.sigmoid(self.fc_hidden(x))
        x = self.fc_out(x)
        x = self.softmax(x)
        return x
