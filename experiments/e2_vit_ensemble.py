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

    from vit.vit_small_16 import VitSmallModel16
    from vit.vit_small_32 import VitSmallModel32
    from vit.vit_medium_32 import VitMediumModel32
    from vit.vit_huge import VitHugeModel
    from vit.vit_ensemble import EnsembleViTModule

    # define the models to ensemble
    vit_small_16 = VitSmallModel16(class_count, device=device)
    model_small_16 = ViTLightningModule.load_from_checkpoint(
        checkpoint_path="checkpoints/vit_small_16/vit_small_16.ckpt",
        vit_model=vit_small_16,
        lr=config.LR,
        loss_fn=nn.CrossEntropyLoss(),
        metrics=metrics,
        scheduler_max_it=config.SCHEDULER_MAX_IT,
    ).vit
        
    vit_small_32 = VitSmallModel32(class_count, device=device)
    model_small_32 = ViTLightningModule.load_from_checkpoint(
        checkpoint_path="checkpoints/vit_small_32/vit_small_32.ckpt",
        vit_model=vit_small_32,
        lr=config.LR,
        loss_fn=nn.CrossEntropyLoss(),
        metrics=metrics,
        scheduler_max_it=config.SCHEDULER_MAX_IT,
    ).vit

    vit_medium_32 = VitMediumModel32(class_count, device=device)
    model_medium_32 = ViTLightningModule.load_from_checkpoint(
        checkpoint_path="checkpoints/vit_medium_32/vit_medium_32.ckpt",
        vit_model=vit_medium_32,
        lr=config.LR,
        loss_fn=nn.CrossEntropyLoss(),
        metrics=metrics,
        scheduler_max_it=config.SCHEDULER_MAX_IT,
    ).vit

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

    ensemble_vit = EnsembleViTModule(
        models=[model_small_16, model_small_32, model_medium_32],
        metrics=metrics,
    )

    # test the ensemble model
    logger_vit_ensemble = WandbLogger(
        project="CR_Leaves", id="vit_ensemble", resume="allow"
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
