import torch
from torch import nn
import torchvision
from torchvision.models import (
    resnet50,
    ResNet50_Weights
)

class ResNet50(nn.Module):
    def __init__(self, num_classes, device) -> None:
        """
        Initializes a Res Net 50 model.

        Args:
            num_classes (int): The number of output classes.
            device (torch.device): The device to run the model on.
        """
        super().__init__()
        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT).to(device=device)

        # freeze the base parameters
        for parameter in self.resnet50.parameters():
            parameter.requires_grad = False

        self.resnet50.fc = torch.nn.Linear(
            in_features=2048, out_features=num_classes
        ).to(device=device)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the Resnet model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.resnet50(x)

class ResNet50(nn.Module):
    def __init__(self, num_classes, device) -> None:
        """
        Initializes a Res Net 50 model.

        Args:
            num_classes (int): The number of output classes.
            device (torch.device): The device to run the model on.
        """
        super().__init__()
        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT).to(device=device)

        # freeze the base parameters
        for parameter in self.resnet50.parameters():
            parameter.requires_grad = False

        self.resnet50.fc = torch.nn.Linear(
            in_features=2048, out_features=num_classes
        ).to(device=device)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the Resnet model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.resnet50(x)