from torch import nn
import torch
from torchvision.models import (
    convnext_base,
    ConvNeXt_Base_Weights
)

class ConvNext(nn.Module):
    def __init__(self, num_classes, device) -> None:
        """
        Initializes a ConvNext model.

        Args:
            num_classes (int): The number of output classes.
            device (torch.device): The device to run the model on.
        """
        super().__init__()
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT).to(
            device=device
        )

        # freeze the base parameters
        for parameter in self.convnext.parameters():
            parameter.requires_grad = False

        self.convnext.classifier[2] = nn.Linear(
            in_features=1024, out_features=num_classes, bias=True
        ).to(device=device)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the ConvNext model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.convnext(x)
