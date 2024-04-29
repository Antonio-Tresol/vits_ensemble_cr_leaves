from pytorch_lightning import LightningModule
from torch import nn
import torch

class EnsembleViTModule(LightningModule):
    """
    LightningModule implementation for an ensemble ViT (Vision Transformer) model.

    Args:
        models (List[LightningModule]): A list of the meodels to ensemble.
        metrics: Metrics used for evaluation.

    Attributes:
        models: A list of the meodels to ensemble.
        loss_fn: Loss function used for training.
        test_metrics: Metrics used for testing evaluation.
    """
    def __init__(
        self,
        models,
        metrics,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.test_metrics = metrics.clone(prefix="test/")
        
    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            None.
        """
        pass

    def forward(self, x):
        """
        Forward pass of the ensemble ViT model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        logits_list = [model(x) for model in self.models]
        logits = torch.stack(logits_list, dim=1).mean(dim=1)
        return logits

    def training_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.argmax(dim=-1)
        self.test_metrics.update(y_hat, y)
        
        return {"train/labels": y, "train/predictions": y_hat}

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)
