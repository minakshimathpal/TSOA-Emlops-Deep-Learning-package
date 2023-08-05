from typing import List, Tuple
from PIL import Image
import hydra
from omegaconf import DictConfig
import lightning.pytorch as L
from lightning.pytorch import LightningModule
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import json

from classifier.utils import pylogger

log = pylogger.get_pylogger(__name__)


#@utils.task_wrapper
def predict(cfg: DictConfig) -> Tuple[dict, dict]:

    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.test_ckpt_path

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    # datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating best model <{cfg.test_ckpt_path}>")
    ckpt = torch.load(cfg.test_ckpt_path)    
    
    log.info(f"Loaded Model: {model}")

    categories = [
        "cat",
        "dog",
    ]

    transforms = T.Compose(
        [
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_path = cfg.test_path
    img = Image.open(image_path)
    if img is None:
        return None
    img = transforms(img).unsqueeze(0)

    log.info("Starting Prediction!")

    logits = model(img)
    preds = F.softmax(logits, dim=1).squeeze(0).tolist()
    out = torch.topk(torch.tensor(preds), 2)
    topk_prob = out[0].tolist()
    topk_label = out[1].tolist()
    print(topk_prob[0])
    print(out)
    print(topk_label[0])
    print()

    print(" \n Top k Predictions :")
    pred_json = {categories[topk_label[i]]: topk_prob[i] for i in range(2)}
    print(json.dumps(pred_json, indent=3))
    print("\n")

    return pred_json


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig):
    # Run inference on the model
    predict(cfg)


if __name__ == "__main__":
    main()