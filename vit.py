from torch import nn
import torch
import torchvision


class VitModel(nn.Module):
    def __init__(self, num_classes, device) -> None:
        super().__init__()
        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        self.vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(
            device
        )

        # freeze the base parameters
        for parameter in self.vit.parameters():
            parameter.requires_grad = False
        self.vit.heads = nn.Linear(in_features=768, out_features=num_classes).to(device)

        self.transforms = pretrained_vit_weights.transforms()

    def forward(self, x) -> torch.Tensor:
        return self.vit(x)

    def get_transforms(self):
        return self.transforms


from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class ViTLightningModule(LightningModule):
    def __init__(self, num_classes, loss_fn, metrics, device):
        super().__init__()
        self.vit = VitModel(num_classes, device)
        self.loss_fn = loss_fn
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, X):
        outputs = self.vit(X)
        return outputs

    def loss(self, preds, ys):
        return self.loss_fn(preds, ys)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        
        y_hat = torch.argmax(torch.softmax(y_hat, dim=-1), dim=-1)
        self.train_metrics.update(y_hat, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "train/labels": y, "train/predictions": y_hat}

    def on_train_epoch_end(self):
        # Compute metrics
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)

        self.train_metrics.reset()

        # train_labels, train_predictions = torch.cat(
        #    [x["train/labels"] for x in outputs], dim=0
        # ), torch.cat([x["train/predictions"] for x in outputs], dim=0)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        y_hat = torch.argmax(torch.softmax(y_hat, dim=-1), dim=-1)
        self.val_metrics.update(y_hat, y)

        self.log("val/loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss, "val/labels": y, "val/predictions": y_hat}

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True)

        self.val_metrics.reset()

        # val_labels, val_predictions = torch.cat(
        #    [x["val/labels"] for x in outputs], dim=0
        # ), torch.cat([x["val/predictions"] for x in outputs], dim=0)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        y_hat = torch.argmax(torch.softmax(y_hat, dim=-1), dim=-1)
        self.test_metrics.update(y_hat, y)

        self.log("test/loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss, "test/labels": y, "test/predictions": y_hat}

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)

        self.test_metrics.reset()

        # test_labels, test_predictions = torch.cat(
        #    [x["test/labels"] for x in outputs], dim=0
        # ), torch.cat([x["test/predictions"] for x in outputs], dim=0)

    def configure_optimizers(self):
        optimizer = AdamW(self.vit.parameters(), lr=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]
