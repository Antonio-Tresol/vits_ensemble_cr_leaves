from torch import nn
import torch
from torchvision.models import (
    efficientnet_b4,
    EfficientNet_B4_Weights
)

class EfficientNetB4(nn.Module):
    def __init__(self, num_classes, device) -> None:
        """
        Initializes a Efficient Net B4 model.

        Args:
            num_classes (int): The number of output classes.
            device (torch.device): The device to run the model on.
        """
        super().__init__()
        self.efficientnet_b4 = efficientnet_b4(
            weights=EfficientNet_B4_Weights.DEFAULT
        ).to(device=device)

        # freeze the base parameters
        for parameter in self.efficientnet_b4.parameters():
            parameter.requires_grad = False

        self.efficientnet_b4.classifier[1] = torch.nn.Linear(
            in_features=1792, out_features=num_classes
        ).to(device=device)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the EfficientNet model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.efficientnet_b4(x)
