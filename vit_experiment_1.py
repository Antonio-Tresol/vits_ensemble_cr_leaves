def main():
    import torch
    from pytorch_lightning.loggers import WandbLogger
    from helper_functions import count_classes

    from vit import ViTLightningModule, get_vit_model_transformations
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping
    from data_modules import CRLeavesDataModule, Sampling
    from torchmetrics.classification import MulticlassAccuracy
    from torchmetrics import MetricCollection
    from torch import nn
    import wandb

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root_dir = "CRLeaves/"
    class_count = count_classes(root_dir)

    metrics = MetricCollection(
        {
            "Accuracy": MulticlassAccuracy(num_classes=class_count, average="micro"),
            "BalancedAccuracy": MulticlassAccuracy(num_classes=class_count),
        }
    )

    model = ViTLightningModule(
        num_classes=class_count,
        loss_fn=nn.CrossEntropyLoss(),
        metrics=metrics,
        device=device,
    )
    train_transform, test_transform = get_vit_model_transformations() 

    cr_leaves_dm = CRLeavesDataModule(

        root_dir=root_dir,
        batch_size=64,
        test_size=0.5,
        use_index=True,
        indices_dir="Indices/",
        sampling=Sampling.NONE,
        train_transform=train_transform,
        test_transform=test_transform,
    )

    cr_leaves_dm.prepare_data()
    cr_leaves_dm.create_data_loaders()

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

    trainer.fit(model, datamodule=cr_leaves_dm)
    trainer.test(model, datamodule=cr_leaves_dm)
    
    wandb.finish()


if __name__ == "__main__":
    main()
