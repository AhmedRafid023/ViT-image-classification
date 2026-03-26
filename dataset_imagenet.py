import torch
import lightning as L
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoImageProcessor

# ==========================================
# Dataset for HF objects (Train Split)
# ==========================================
class HFTinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, processor):
        self.dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        label = item['label']
        
        inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# Dataset for Local CSVs (Val / Test Splits)
# ==========================================
class CSVImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, processor):
        self.data = pd.read_csv(csv_path)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        image = Image.open(image_path).convert('RGB')
        label = row['label']
        
        inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "image_path": image_path  # <-- Essential for the test.py output CSV
        }

# ==========================================
# Lightning DataModule
# ==========================================
class TinyImageNetDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_id = config['model']['id']
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Load 100k Train directly from Hugging Face memory
            hf_train = load_dataset("zh-plus/tiny-imagenet", split="train")
            self.train_ds = HFTinyImageNetDataset(hf_train, self.processor)
            
            # Load 9k Val from the local CSV you generated
            self.val_ds = CSVImageDataset(self.config['data']['val_csv'], self.processor)
            print(f"Train: {len(self.train_ds)} (HF) | Val: {len(self.val_ds)} (Local)")
            
        if stage == 'test':
            # Load 1k Test from the local CSV
            self.test_ds = CSVImageDataset(self.config['test']['test_csv'], self.processor)
            print(f"Test: {len(self.test_ds)} (Local)")

    def train_dataloader(self):
        # We use .get() to safely handle differences between train.yaml and test.yaml keys
        batch_size = self.config['data'].get('train_batch_size', self.config['data'].get('batch_size', 256))
        return DataLoader(
            self.train_ds, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=self.config['data']['num_workers']
        )

    def val_dataloader(self):
        batch_size = self.config['data'].get('val_batch_size', self.config['data'].get('batch_size', 128))
        return DataLoader(
            self.val_ds, 
            batch_size=batch_size, 
            num_workers=self.config['data']['num_workers']
        )

    def test_dataloader(self):
        batch_size = self.config['data'].get('batch_size', 128)
        return DataLoader(
            self.test_ds, 
            batch_size=batch_size, 
            num_workers=self.config['data']['num_workers']
        )