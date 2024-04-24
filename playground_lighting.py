def main():
    import torch
    from pytorch_lightning.loggers import WandbLogger
    from helper_functions import count_classes
    
    from vit import ViTLightningModule
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping
    from data_modules import CRLeavesDataModule, Sampling
    from torchmetrics.classification import MulticlassAccuracy
    from torchmetrics import MetricCollection
    from torch import nn
    from metrics import MRR

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root_dir = "CRLeaves/"
    class_count = count_classes(root_dir)

    metrics = MetricCollection(
        {
            "Accuracy": MulticlassAccuracy(num_classes=class_count, average="micro"),
            "BalancedAccuracy": MulticlassAccuracy(num_classes=class_count)
        }
    )
    
    model = ViTLightningModule(
        num_classes=class_count,
        loss_fn=nn.CrossEntropyLoss(),
        metrics=metrics,
        device=device,
    )

    data_module = CRLeavesDataModule(
        root_dir=root_dir,
        batch_size=32,
        test_size=0.2,
        use_index=True,
        indices_dir="Indices/",
        sampling=Sampling.NONE,
        train_transform=model.vit.get_transforms(),
        test_transform=model.vit.get_transforms(),
    )

    data_module.prepare_data()
    data_module.create_data_loaders()

    # for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=3, strict=False, verbose=False, mode="min"
    )

    wandb_logger = WandbLogger(project="CR_Leaves", resume="allow")

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=early_stop_callback,
        max_epochs=30,
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=data_module)
    
