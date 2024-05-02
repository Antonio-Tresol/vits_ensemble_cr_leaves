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

    from conv.conv_module import get_conv_model_transformations
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

    from conv.conv_ensemble import EnsembleConvModule
    from conv.convnext import ConvNext
    from conv.efficientnet import EfficientNetB4
    from conv.resnet import ResNet50
    from conv.conv_module import ConvolutionalLightningModule

    # prepare the data
    train_transform, test_transform = get_conv_model_transformations()
    cr_leaves_dm = CRLeavesDataModule(
        root_dir=config.ROOT_DIR,
        batch_size=config.BATCH_SIZE,
        test_size=config.TEST_SIZE,
        use_index=config.USE_INDEX,
        indices_dir=config.INDICES_DIR,
        sampling=Sampling.NONE,
        train_transform=test_transform,
        test_transform=test_transform,
    )

    cr_leaves_dm.prepare_data()
    cr_leaves_dm.create_data_loaders()

    metrics_data = []
    for i in range(config.NUM_TRIALS):
        # define the models to ensemble
        checkpoint_path = config.RESNET_DIR + config.RESNET_FILENAME + str(i) + ".ckpt"
        resnet = ResNet50(class_count, device=device)
        resnet = ConvolutionalLightningModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            conv_model=resnet,
            lr=config.LR,
            loss_fn=nn.CrossEntropyLoss(),
            metrics=metrics,
            scheduler_max_it=config.SCHEDULER_MAX_IT,
        ).model

        checkpoint_path = (
            config.CONVNEXT_DIR + config.CONVNEXT_FILENAME + str(i) + ".ckpt"
        )
        convnet = ConvNext(class_count, device=device)
        convnet = ConvolutionalLightningModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            conv_model=convnet,
            lr=config.LR,
            loss_fn=nn.CrossEntropyLoss(),
            metrics=metrics,
            scheduler_max_it=config.SCHEDULER_MAX_IT,
        ).model

        checkpoint_path = (
            config.EFFICIENTNET_DIR + config.EFFICIENTNET_FILENAME + str(i) + ".ckpt"
        )
        efficientnet = EfficientNetB4(class_count, device=device)
        efficientnet = ConvolutionalLightningModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            conv_model=efficientnet,
            lr=config.LR,
            loss_fn=nn.CrossEntropyLoss(),
            metrics=metrics,
            scheduler_max_it=config.SCHEDULER_MAX_IT,
        ).model

        ensemble_conv = EnsembleConvModule(
            models=[resnet, convnet, efficientnet],
            metrics=metrics,
        )

        id = config.CONV_ENSEMBLE_FILENAME + str(i) + "_" + wandb.util.generate_id()
        # test the ensemble model
        logger_conv_ensemble = WandbLogger(
            project=config.WANDB_PROJECT, id=id, resume="allow"
        )

        trainer_ensemble = Trainer(
            logger=logger_conv_ensemble,
            max_epochs=config.EPOCHS,
            log_every_n_steps=1,
        )

        metrics_data.append(
            trainer_ensemble.test(ensemble_conv, datamodule=cr_leaves_dm)[0]
        )
        wandb.finish()

    pd.DataFrame(metrics_data).to_csv(config.CONV_ENSEMBLE_CSV_FILENAME, index=False)


if __name__ == "__main__":
    main()
