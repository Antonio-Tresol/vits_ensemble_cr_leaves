import torch
from vit_small import VitSmallModel
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


class ViTLightningModule(LightningModule):
    """
    LightningModule implementation for the ViT (Vision Transformer) model.

    Args:
        num_classes (int): Number of output classes.
        loss_fn: Loss function used for training.
        metrics: Metrics used for evaluation.
        device: Device to run the model on.
        lr (float): Learning rate for the optimizer.
        scheduler_max_it (int): Maximum number of iterations for the learning rate scheduler.
        weight_decay (float): Weight decay for the optimizer.
        epsilon (float): Epsilon value for numerical stability.

    Attributes:
        vit: ViTModel instance representing the Vision Transformer model.
        loss_fn: Loss function used for training.
        train_metrics: Metrics used for training evaluation.
        val_metrics: Metrics used for validation evaluation.
        test_metrics: Metrics used for testing evaluation.
        lr (float): Learning rate for the optimizer.
        scheduler_max_it (int): Maximum number of iterations for the learning rate scheduler.
    """

    def __init__(
        self, vit_model,
        loss_fn,
        metrics,
        lr,
        scheduler_max_it,
        weight_decay=0,
    ):
        super().__init__()
        self.vit = vit_model 
        self.loss_fn = loss_fn
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        self.lr = lr
        self.scheduler_max_it = scheduler_max_it
        self.weight_decay = weight_decay

    def forward(self, X):
        """
        Forward pass of the ViT model.

        Args:
            X: Input tensor.

        Returns:
            Output tensor.
        """
        outputs = self.vit(X)
        return outputs

    def _common_step(self, batch, batch_idx):
        """
        Common step for training, validation, and testing.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Tuple containing the ground truth labels, predicted outputs, and loss value.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return y, y_hat, loss

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing the loss value, ground truth labels, and predicted outputs.
        """
        y, y_hat, loss = self._common_step(batch, batch_idx)

        y_hat = torch.argmax(torch.softmax(y_hat, dim=-1), dim=-1)
        self.train_metrics.update(y_hat, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "train/labels": y, "train/predictions": y_hat}

    def on_train_epoch_end(self):
        """
        Callback function called at the end of each training epoch.
        Computes and logs the training metrics.
        """
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)

        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing the loss value, ground truth labels, and predicted outputs.
        """
        y, y_hat, loss = self._common_step(batch, batch_idx)

        y_hat = torch.argmax(torch.softmax(y_hat, dim=-1), dim=-1)
        self.val_metrics.update(y_hat, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "val/labels": y, "val/predictions": y_hat}

    def on_validation_epoch_end(self):
        """
        Callback function called at the end of each validation epoch.
        Computes and logs the validation metrics.
        """
        self.log_dict(
            self.val_metrics.compute(), on_step=False, on_epoch=True, prog_bar=True
        )

        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing the loss value, ground truth labels, and predicted outputs.
        """
        y, y_hat, loss = self._common_step(batch, batch_idx)

        y_hat = torch.argmax(torch.softmax(y_hat, dim=-1), dim=-1)
        self.test_metrics.update(y_hat, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "test/labels": y, "test/predictions": y_hat}

    def on_test_epoch_end(self):
        """
        Callback function called at the end of each testing epoch.
        Computes and logs the testing metrics.
        """
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)

        self.test_metrics.reset()

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Tuple containing the optimizer and learning rate scheduler.
        """
        optimizer = Adam(self.vit.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.scheduler_max_it)
        return [optimizer], [scheduler]


from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_vit_model_transformations() -> tuple[transforms.Compose]:
    """
    Returns the train and test transformations for the VIT model.
    Values are taking from the tranformation originally made by the VIT model.

    Returns:
        tuple[transforms.Compose]: A tuple containing the train and test transformations.
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize(255, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomCrop(223),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(44),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[-1.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(255, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(223),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[-1.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return train_transform, test_transform
