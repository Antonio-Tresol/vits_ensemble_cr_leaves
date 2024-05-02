def main():
    import os
    import sys
    import inspect
    import os

    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    currentdir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)

    import torch
    from pytorch_lightning.loggers import WandbLogger
    from helper_functions import count_classes
    import pandas as pd
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
    
    root_dir = "CRLeaves/"
    class_count = count_classes(root_dir)

    metrics = MetricCollection(
        {
            "Accuracy": MulticlassAccuracy(num_classes=class_count, average="micro"),
            "BalancedAccuracy": MulticlassAccuracy(num_classes=class_count),
        }
    )
    from vit.deit3_base_16 import Deit3Base16

    deit_base_16 = Deit3Base16(class_count, device=device)

    test_transform = deit_base_16.get_transforms()

    cr_leaves_dm = CRLeavesDataModule(
        root_dir=root_dir,
        batch_size=config.BATCH_SIZE,
        test_size=config.TEST_SIZE,
        use_index=config.USE_INDEX,
        indices_dir="Indices/",
        sampling=Sampling.NONE,
        train_transform=test_transform,
        test_transform=test_transform,
    )

    cr_leaves_dm.prepare_data()
    cr_leaves_dm.create_data_loaders()
    metrics_data = []
    for i in range(config.NUM_TRIALS):
        deit_base_16 = Deit3Base16(class_count, device=device)
        early_stop_callback = EarlyStopping(
            monitor="val/loss",
            patience=config.PATIENCE,
            strict=False,
            verbose=False,
            mode="min",
        )
        model = ViTLightningModule(
            vit_model=deit_base_16,
            lr=config.LR,
            loss_fn=nn.CrossEntropyLoss(),
            metrics=metrics,
            scheduler_max_it=config.SCHEDULER_MAX_IT,
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            dirpath=config.DEIT3_BASE_16_DIR,
            filename=config.DEIT3_BASE_16_FILENAME + str(i),
            save_top_k=config.TOP_K_SAVES,
            mode="min",
        )

        id = config.DEIT3_BASE_16_FILENAME + str(i) + "_" + wandb.util.generate_id()
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
    pd.DataFrame(metrics_data).to_csv(
        config.DEIT3_BASE_16_CSV_FILENAME, index=False
    )


if __name__ == "__main__":
    main()
