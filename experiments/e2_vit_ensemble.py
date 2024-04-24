def main():
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)

    import torch
    from pytorch_lightning.loggers import WandbLogger
    from helper_functions import count_classes

    from vit.vit_module import ViTLightningModule, get_vit_model_transformations
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelSummary
    from data_modules import CRLeavesDataModule, Sampling
    from torchmetrics.classification import MulticlassAccuracy
    from torchmetrics import MetricCollection
    from torch import nn
    import wandb
    import configuration as config

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
    from vit.vit_small import VitSmallModel
    from vit.vit_medium import VitMediumModel
    from vit.vit_huge import VitHugeModel 
    vit_small = VitSmallModel(class_count, device=device) 
    model_small = ViTLightningModule(
        vit_model=vit_small,
        loss_fn=nn.CrossEntropyLoss(),
        metrics=metrics,
        lr=config.LR,
        scheduler_max_it=config.SCHEDULER_MAX_IT,
    )
    vit_medium = VitMediumModel(class_count, device=device)
    model_medium = ViTLightningModule(
        vit_model=vit_small,
        loss_fn=nn.CrossEntropyLoss(),
        metrics=metrics,
        lr=config.LR,
        scheduler_max_it=config.SCHEDULER_MAX_IT,
    )
    vit_huge = VitMediumModel(class_count, device=device)
    model_large = ViTLightningModule(
        vit_model=vit_small,
        loss_fn=nn.CrossEntropyLoss(),
        metrics=metrics,
        lr=config.LR,
        scheduler_max_it=config.SCHEDULER_MAX_IT,
    )

    train_transform, test_transform = get_vit_model_transformations()

    cr_leaves_dm = CRLeavesDataModule(
        root_dir=root_dir,
        batch_size=config.BATCH_SIZE,
        test_size=config.TEST_SIZE,
        use_index=True,
        indices_dir="Indices/",
        sampling=Sampling.NONE,
        train_transform=train_transform,
        test_transform=test_transform,
    )

    cr_leaves_dm.prepare_data()
    cr_leaves_dm.create_data_loaders()

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=config.PATIENCE,
        strict=False,
        verbose=False,
        mode="min",
    )

    wandb_logger = WandbLogger(project="CR_Leaves", id="vit_small" resume="allow")

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=early_stop_callback,
        max_epochs=config.EPOCHS,
        log_every_n_steps=1,
    )

    trainer.fit(model_small, datamodule=cr_leaves_dm)
    trainer.test(model_small, datamodule=cr_leaves_dm)

    wandb.finish()


if __name__ == "__main__":
    main()
