def main():
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    import torch
    from pytorch_lightning.loggers import WandbLogger
    from helper_functions import count_classes

    from conv.conv_module import (
        ConvolutionalLightningModule,
        get_conv_model_transformations,
    )
    from pytorch_lightning.callbacks import ModelCheckpoint
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

    class_count = count_classes(config.ROOT_DIR)

    metrics = MetricCollection(
        {
            "Accuracy": MulticlassAccuracy(num_classes=class_count, average="micro"),
            "BalancedAccuracy": MulticlassAccuracy(num_classes=class_count),
        }
    )
    from conv.efficientnet import EfficientNetB4

    efficientNet = EfficientNetB4(num_classes=class_count, device=device)
    model = ConvolutionalLightningModule(
        conv_model=efficientNet,
        loss_fn=nn.CrossEntropyLoss(),
        metrics=metrics,
        lr=config.LR,
        scheduler_max_it=config.SCHEDULER_MAX_IT,
    )

    train_transform, test_transform = get_conv_model_transformations()

    cr_leaves_dm = CRLeavesDataModule(
        root_dir=config.ROOT_DIR,
        batch_size=config.BATCH_SIZE,
        test_size=config.TEST_SIZE,
        use_index=True,
        indices_dir=config.INDICES_DIR,
        sampling=Sampling.NONE,
        train_transform=test_transform,
        test_transform=test_transform,
    )

    cr_leaves_dm.prepare_data()
    cr_leaves_dm.create_data_loaders()

    for i in range(config.NUM_TRIALS):

        early_stop_callback = EarlyStopping(
            monitor="val/loss",
            patience=config.PATIENCE,
            strict=False,
            verbose=False,
            mode="min",
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            dirpath=config.EFFICIENTNET_DIR,
            filename=config.EFFICIENTNET_FILENAME + str(i),
            save_top_k=config.TOP_K_SAVES,
            mode="min",
        )

        id = config.EFFICIENTNET_FILENAME + str(i)
        wandb_logger = WandbLogger(project=config.WAND_PROJECT, id=id, resume="allow")

        trainer = Trainer(
            logger=wandb_logger,
            callbacks=[early_stop_callback, checkpoint_callback],
            max_epochs=config.EPOCHS,
            log_every_n_steps=1,
        )

        trainer.fit(model, datamodule=cr_leaves_dm)
        trainer.test(model, datamodule=cr_leaves_dm)

    wandb.finish()


if __name__ == "__main__":
    main()
