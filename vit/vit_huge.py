from torch import nn
import torchvision
import torch

class VitHugeModel(nn.Module):
    def __init__(self, num_classes, device) -> None:
        """
        Initializes a Vision Transformer (ViT) biggest model.

        Args:
            num_classes (int): The number of output classes.
            device (torch.device): The device to run the model on.
        """
        super().__init__()
        pretrained_vit_weights = torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1
        self.vit = torchvision.models.vit_h_14(weights=pretrained_vit_weights).to(
            device
        )

        # freeze the base parameters
        for parameter in self.vit.parameters():
            parameter.requires_grad = False
        self.vit.heads = nn.Linear(in_features=1280, out_features=num_classes).to(device)

        self.transforms = pretrained_vit_weights.transforms()

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the ViT model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.vit(x)