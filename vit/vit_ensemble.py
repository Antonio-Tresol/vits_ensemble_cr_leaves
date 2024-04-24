from pytorch_lightning import LightningModule
from torch import nn
import torch
class EnsembleViTModule(LightningModule):
    def __init__(
        self,
        models,
        class_count,
        species_count,
        metrics,
        confusion_matrix = None
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.class_count = class_count
        self.species_count = species_count
        self.test_metrics = metrics.clone(prefix="test/")
        self.cm = confusion_matrix
        
    def configure_optimizers(self):
        pass

    def forward(self, x):
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
        self.test_cm.update(y_hat, y)

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)
        cm = self.log.plot.confusion_matrix(
            y_true=ground_truth,
            preds=predictions,
            class_names=class_names)
            
        wandb.log({"conf_mat": cm})
    
    def on_test_end(self):


