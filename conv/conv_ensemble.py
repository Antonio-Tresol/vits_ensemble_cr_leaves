from pytorch_lightning import LightningModule
from torch import nn
import torch

class EnsembleViTModule(LightningModule):
    """
    LightningModule implementation for an ensemble Convolution (Vision Transformer) model.

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
        """
        Training step.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            None.
        """
        pass

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Args:
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary containing the loss value, ground truth labels, and predicted outputs.
        """
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.argmax(dim=-1)
        self.test_metrics.update(y_hat, y)
        
        cm = self.log.plot.confusion_matrix(
            y_true=y,
            preds=y_hat)
            
        self.log({"test/confusion_matrix": cm})
        
    def on_test_epoch_end(self):
        """
        Callback function called at the end of each testing epoch.
        Computes and logs the testing metrics.
        """
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)
