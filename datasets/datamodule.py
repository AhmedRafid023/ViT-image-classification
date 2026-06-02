"""
Lightning DataModule wrapper for any dassl DatasetBase subclass.

Usage in train.py:
    from dassl.data.datasets import DATASET_REGISTRY
    from datasets.datamodule import DatasetModuleWrapper

    dataset_cls = DATASET_REGISTRY.get(dataset_name)
    dm = DatasetModuleWrapper(dataset_cls, config)
"""

import torch
import lightning as L
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor


# ---------------------------------------------------------------------------
# CfgAdapter — translates our plain YAML dict into dassl-style cfg.DATASET.* access
# ---------------------------------------------------------------------------

class CfgAdapter:
    """Wraps a plain config dict so dassl DatasetBase subclasses can use cfg.DATASET.ROOT etc."""

    class _DatasetSection:
        def __init__(self, config):
            self.ROOT = config["dataset"].get("root", "./data")
            self.NUM_SHOTS = 0       # always use full dataset
            self.SUBSAMPLE_CLASSES = "all"  # always use all classes

    def __init__(self, config):
        self.DATASET = self._DatasetSection(config)
        self.SEED = config.get("seed", 0)


# ---------------------------------------------------------------------------
# DatumDataset — wraps a list of dassl Datum objects as a PyTorch Dataset
# ---------------------------------------------------------------------------

class DatumDataset(Dataset):
    def __init__(self, items, processor):
        self.items = items
        self.processor = processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item.impath).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(item.label, dtype=torch.long),
            "image_path": item.impath,
        }


# ---------------------------------------------------------------------------
# DatasetModuleWrapper — Lightning DataModule for any dassl DatasetBase subclass
# ---------------------------------------------------------------------------

class DatasetModuleWrapper(L.LightningDataModule):
    def __init__(self, dataset_cls, config):
        super().__init__()
        self.dataset_cls = dataset_cls
        self.config = config
        self.processor = AutoImageProcessor.from_pretrained(config["model"]["id"])

    def _build_dassl_dataset(self):
        if not hasattr(self, "_dassl_dataset"):
            self._dassl_dataset = self.dataset_cls(CfgAdapter(self.config))
        return self._dassl_dataset

    def get_classnames(self):
        """Return list of classnames indexed by label (derived from train split)."""
        dataset = self._build_dassl_dataset()
        classnames_by_label = {}
        for item in dataset.train_x + dataset.val + dataset.test:
            classnames_by_label[item.label] = item.classname
        return [classnames_by_label[i] for i in sorted(classnames_by_label.keys())]

    def setup(self, stage=None):
        dataset = self._build_dassl_dataset()

        if stage in ("fit", None):
            self.train_ds = DatumDataset(dataset.train_x, self.processor)
            self.val_ds = DatumDataset(dataset.val, self.processor)
            print(f"Train: {len(self.train_ds)} | Val: {len(self.val_ds)}")

        if stage == "test":
            self.test_ds = DatumDataset(dataset.test, self.processor)
            print(f"Test: {len(self.test_ds)}")

    def train_dataloader(self):
        bs = self.config["data"].get("train_batch_size", self.config["data"].get("batch_size", 64))
        nw = self.config["data"].get("num_workers", 4)
        return DataLoader(self.train_ds, batch_size=bs, shuffle=True, num_workers=nw)

    def val_dataloader(self):
        bs = self.config["data"].get("val_batch_size", self.config["data"].get("batch_size", 64))
        nw = self.config["data"].get("num_workers", 4)
        return DataLoader(self.val_ds, batch_size=bs, num_workers=nw)

    def test_dataloader(self):
        bs = self.config["data"].get("val_batch_size", self.config["data"].get("batch_size", 64))
        nw = self.config["data"].get("num_workers", 4)
        return DataLoader(self.test_ds, batch_size=bs, num_workers=nw)