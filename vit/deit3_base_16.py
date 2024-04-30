import timm
from torch import nn


class Deit3Base16(nn.Module):
    def __init__(self, num_classes, device):
        super().__init__()
        self.vit = timm.create_model(
            "deit3_base_patch16_224.fb_in1k", pretrained=True, num_classes=num_classes
        ).to(device)
        # freeze the model
        for param in self.vit.parameters():
            param.requires_grad = False

        # change the last layer
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes).to(device)

    def forward(self, x):
        return self.vit(x)

    def get_transforms(self):
        return timm.data.create_transform(
            **timm.data.resolve_model_data_config(self.vit)
        )
