from typing import Tuple, Dict

import lightning as L
import torch
import hydra
from omegaconf import DictConfig
import sys
sys.path.append("./")
from omegaconf import DictConfig, OmegaConf
from classifier.utils.pylogger import get_pylogger
from classifier import utils
from lightning.pytorch.tuner import Tuner

log = get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """
    Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
       
    
    1) here we are downloading data by calling hydra.utils.instantiate(cfg.data)
    2) instantiating the model using hydra.utils.instantiate(cfg.model)
    3) instantiating the lightining model hydra.utils.instantiate(cfg.trainer)
       
    """
    # set seed for random number generators in pytorch, numpy and python.random

    print(OmegaConf.to_yaml(cfg))
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
        "callbacks": callbacks,
        "logger": logger,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        model = torch.compile(model)

    if cfg.get("tuner"):
        log.info("Running LR Finder!")
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, datamodule)
        print(f"best initial lr={lr_finder.suggestion()}")
        # model.hparams.learning_rate = lr_finder.suggestion()

        log.info("Running Batch Size Finder!")
        # Auto-scale batch size by growing it exponentially (default)
        tuner.scale_batch_size(model, datamodule, mode="power")
        print(f"optimal batch size = {datamodule.hparams.batch_size}")

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

        # ckpt_save_path = cfg.get('ckpt_save_path')
        # trainer.save_checkpoint(ckpt_save_path)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path # UPDATE ! now we can get the best model from the callbacks
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # train the model
    metric_dict, _ = train(cfg)

    # this will be used by hydra later for optimization
    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()