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
    # prepare the data
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

    from vit.vit_base_16 import VitBase16
    from vit.vit_base_32 import VitBase32
    from vit.vit_ensemble import EnsembleViTModule
    from vit.deit3_base_16 import Deit3Base16

    for i in range(config.NUM_TRIALS):
        # define the models to ensemble
        vit_base_16 = VitBase16(class_count, device=device)
        checkpoint_filename = (
            config.VIT_BASE_16_DIR + config.VIT_BASE_16_FILENAME + str(i) + ".ckpt"
        )
        vit_base_16 = ViTLightningModule.load_from_checkpoint(
            checkpoint_path=checkpoint_filename,
            vit_model=vit_base_16,
            lr=config.LR,
            loss_fn=nn.CrossEntropyLoss(),
            metrics=metrics,
            scheduler_max_it=config.SCHEDULER_MAX_IT,
        ).vit

        deit_base_16 = Deit3Base16(class_count, device=device)
        checkpoint_filename = (
            config.DEIT3_BASE_16_DIR + config.DEIT3_BASE_16_FILENAME + str(i) + ".ckpt"
        )
        deit_base_16 = ViTLightningModule.load_from_checkpoint(
            checkpoint_path=checkpoint_filename,
            vit_model=deit_base_16,
            lr=config.LR,
            loss_fn=nn.CrossEntropyLoss(),
            metrics=metrics,
            scheduler_max_it=config.SCHEDULER_MAX_IT,
        ).vit

        checkpoint_filename = (
            config.VIT_BASE_32_DIR + config.VIT_BASE_32_FILENAME + str(i) + ".ckpt"
        )
        vit_base_32 = VitBase32(class_count, device=device)
        vit_base_32 = ViTLightningModule.load_from_checkpoint(
            checkpoint_path=checkpoint_filename,
            vit_model=vit_base_32,
            lr=config.LR,
            loss_fn=nn.CrossEntropyLoss(),
            metrics=metrics,
            scheduler_max_it=config.SCHEDULER_MAX_IT,
        ).vit

        ensemble_vit = EnsembleViTModule(
            models=[vit_base_16, deit_base_16, vit_base_32],
            metrics=metrics,
        )

        id = "vit_ensemble_" + str(i)

        # test the ensemble model
        logger_vit_ensemble = WandbLogger(
            project=config.WANDB_PROJECT, id=id, resume="allow"
        )
        trainer_ensemble = Trainer(
            logger=logger_vit_ensemble,
            max_epochs=config.EPOCHS,
            log_every_n_steps=1,
        )

        trainer_ensemble.test(ensemble_vit, datamodule=cr_leaves_dm)
        wandb.finish()


if __name__ == "__main__":
    main()
