import torch
import torchvision
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

class ConvolutionalLightningModule(LightningModule):
    """
    LightningModule implementation for the ViT (Vision Transformer) model.

    Args:
        conv_model: Model instance representing the Convolution Network Model.
        loss_fn: Loss function used for training.
        metrics: Metrics used for evaluation.
        lr (float): Learning rate for the optimizer.
        scheduler_max_it (int): Maximum number of iterations for the learning rate scheduler.
        weight_decay (float): Weight decay for the optimizer.

    Attributes:
        model: Model instance representing the Convolution Network Model.
        loss_fn: Loss function used for training.
        train_metrics: Metrics used for training evaluation.
        val_metrics: Metrics used for validation evaluation.
        test_metrics: Metrics used for testing evaluation.
        lr (float): Learning rate for the optimizer.
        scheduler_max_it (int): Maximum number of iterations for the learning rate scheduler.
    """
    def __init__(
        self,
        conv_model,
        loss_fn,
        metrics,
        lr,
        scheduler_max_it,
        weight_decay=0,
    ):
        super().__init__()
        self.model = conv_model
        self.loss_fn = loss_fn
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        self.lr = lr
        self.scheduler_max_it = scheduler_max_it
        self.weight_decay = weight_decay

    def forward(self, X):
        """
        Forward pass of the Convolutional model.

        Args:
            X: Input tensor.

        Returns:
            Output tensor.
        """
        outputs = self.model(X)
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
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.scheduler_max_it)
        return [optimizer], [scheduler]


def get_conv_model_transformations() -> tuple[torchvision.transforms.Compose]:
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(232),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomRotation(45),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )

    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(232),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )
    return train_transform, test_transform
