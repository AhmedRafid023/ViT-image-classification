import os
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from dotenv import load_dotenv

import datasets as _datasets  # noqa: F401 — registers all dataset classes
from datasets import DATASET_REGISTRY
from datasets.datamodule import DatasetModuleWrapper
from models.vitllava import ProjectorAblationModel
from utils.tools import update_results

load_dotenv()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    mode   = config["mode"]

    if mode == "train":
        _train(cfg, config)
    elif mode == "test":
        _test(config)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use mode=train or mode=test.")


def _train(cfg, config):
    choices      = HydraConfig.get().runtime.choices
    dataset_name = config["dataset"]["name"]
    sit          = config["experiment"]["situation"]

    print(f"Training: {choices.model} | Dataset: {dataset_name} | Sit: {sit}")

    dm    = DatasetModuleWrapper(DATASET_REGISTRY.get(dataset_name), config)
    model = ProjectorAblationModel(config)

    run = wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        name=f"{choices.dataset}/{choices.model}/{sit}",
        config=config,
        mode=config["wandb"]["mode"],
        dir="outputs",
    )

    OmegaConf.save(cfg, f"{run.dir}/config.yaml", resolve=True)
    config["test"]["output_csv"] = f"{run.dir}/{dataset_name}_results.csv"

    checkpoint_callback = ModelCheckpoint(
        dirpath=run.dir,
        filename=f"{sit}-{{epoch:02d}}",
        save_top_k=1,
        monitor="val_acc",
        mode="max",
        save_last=True,
    )

    trainer = L.Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
        precision=config["trainer"]["precision"],
        logger=WandbLogger(experiment=run),
    )

    trainer.fit(model, datamodule=dm)

    if config["test"]["after_train"]:
        model.classnames = dm.get_classnames()
        model.test_meta  = {"model": choices.model, "dataset": choices.dataset}
        trainer.test(model, datamodule=dm, ckpt_path="best")

        for ckpt in [checkpoint_callback.best_model_path, checkpoint_callback.last_model_path]:
            if ckpt and os.path.exists(ckpt):
                os.remove(ckpt)


def _test(config):
    choices   = HydraConfig.get().runtime.choices
    ckpt_path = config["test"]["checkpoint"]
    if ckpt_path is None:
        raise ValueError("Checkpoint path is required. Pass test.checkpoint=<path>")

    dataset_name = config["dataset"]["name"]
    out_csv      = config["test"]["output_csv"] or os.path.join(
        os.path.dirname(os.path.abspath(ckpt_path)), f"{dataset_name}_results.csv"
    )

    print(f"Testing Checkpoint: {ckpt_path}")

    dm         = DatasetModuleWrapper(DATASET_REGISTRY.get(dataset_name), config)
    classnames = dm.get_classnames()
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ProjectorAblationModel.load_from_checkpoint(ckpt_path, config=config)
    model.to(device)
    model.eval()

    results    = []
    all_preds  = []
    all_labels = []

    print(f"\n--- Running Inference on {len(dm.test_ds)} Test Images ---")

    autocast_dtype = torch.float16 if device.type == "cuda" else torch.float32

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
        for batch in tqdm(test_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels       = batch["label"].to(device)
            paths        = batch.get("image_path", ["unknown"] * len(labels))

            logits = model(pixel_values)
            preds  = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(len(labels)):
                true_idx = labels[i].item()
                pred_idx = preds[i].item()
                row = {
                    "image_path":     paths[i],
                    "true_label_idx": true_idx,
                    "pred_label_idx": pred_idx,
                    "is_correct":     int(true_idx == pred_idx),
                }
                if classnames:
                    row["true_label_name"] = classnames[true_idx] if true_idx < len(classnames) else str(true_idx)
                    row["pred_label_name"] = classnames[pred_idx] if pred_idx < len(classnames) else str(pred_idx)
                results.append(row)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nDetailed results saved to: {out_csv}")

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy   = (all_preds == all_labels).mean()
    print("=" * 40)
    print(f"FINAL TEST ACCURACY: {accuracy * 100:.2f}%")
    print("=" * 40)

    update_results(
        choices.model,
        choices.dataset,
        config["experiment"]["situation"],
        round(accuracy * 100, 2),
        projector_type=config["model"].get("projector_type", ""),
        bridge_type=config["model"].get("bridge_type", ""),
    )


if __name__ == "__main__":
    main()
