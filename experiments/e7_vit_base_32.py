def main():
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)

    import pandas as pd
    import torch
    from pytorch_lightning.loggers import WandbLogger
    from helper_functions import count_classes

    from pytorch_lightning.callbacks import ModelCheckpoint
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

    class_count = count_classes(config.ROOT_DIR)

    metrics = MetricCollection(
        {
            "Accuracy": MulticlassAccuracy(num_classes=class_count, average="micro"),
            "BalancedAccuracy": MulticlassAccuracy(num_classes=class_count),
        }
    )

    train_transform, test_transform = get_vit_model_transformations()

    cr_leaves_dm = CRLeavesDataModule(
        root_dir=config.ROOT_DIR,
        batch_size=config.BATCH_SIZE,
        test_size=config.TEST_SIZE,
        use_index=USE_INDEX,
        indices_dir=config.INDICES_DIR,
        sampling=Sampling.NONE,
        train_transform=train_transform,
        test_transform=test_transform,
    )

    cr_leaves_dm.prepare_data()
    cr_leaves_dm.create_data_loaders()

    from vit.vit_base_32 import VitBase32

    metrics_data = []
    for i in range(config.NUM_TRIALS):

        vit_small = VitBase32(class_count, device=device)
        model = ViTLightningModule(
            vit_model=vit_small,
            loss_fn=nn.CrossEntropyLoss(),
            metrics=metrics,
            lr=config.LR,
            scheduler_max_it=config.SCHEDULER_MAX_IT,
        )

        early_stop_callback = EarlyStopping(
            monitor="val/loss",
            patience=config.PATIENCE,
            strict=False,
            verbose=False,
            mode="min",
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            dirpath=config.VIT_BASE_32_DIR,
            filename=config.VIT_BASE_32_FILENAME + str(i),
            save_top_k=config.TOP_K_SAVES,
            mode="min",
        )

        id = config.VIT_BASE_32_FILENAME + str(i)

        wandb_logger = WandbLogger(project=config.WANDB_PROJECT, id=id, resume="allow")

        trainer = Trainer(
            logger=wandb_logger,
            callbacks=[early_stop_callback, checkpoint_callback],
            max_epochs=config.EPOCHS,
            log_every_n_steps=1,
        )

        trainer.fit(model, datamodule=cr_leaves_dm)
        metrics_data.append(trainer.test(model, datamodule=cr_leaves_dm)[0])
        wandb.finish()

    pd.DataFrame(metrics_data).to_csv(config.VIT_BASE_32_CSV_FILENAME, index=False)


if __name__ == "__main__":
    main()
