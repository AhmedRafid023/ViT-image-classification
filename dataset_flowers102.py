import torch
import lightning as L
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

# ==========================================
# Dataset for Flowers102
# ==========================================
class HFProcessorFlowersDataset(torch.utils.data.Dataset):
    def __init__(self, torchvision_dataset, processor):
        self.dataset = torchvision_dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Torchvision Flowers102 returns PIL Image
        
        inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# Lightning DataModule
# ==========================================
class Flowers102DataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_id = config['model']['id']
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.data_dir = self.config['data'].get('data_dir', './data')

    def prepare_data(self):
        # Download data if needed
        Flowers102(root=self.data_dir, split="train", download=True)
        Flowers102(root=self.data_dir, split="val", download=True)
        Flowers102(root=self.data_dir, split="test", download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            raw_train = Flowers102(root=self.data_dir, split="train", download=False)
            raw_val = Flowers102(root=self.data_dir, split="val", download=False)
            
            self.train_ds = HFProcessorFlowersDataset(raw_train, self.processor)
            self.val_ds = HFProcessorFlowersDataset(raw_val, self.processor)
            print(f"Train: {len(self.train_ds)} | Val: {len(self.val_ds)}")
            
        if stage == 'test':
            raw_test = Flowers102(root=self.data_dir, split="test", download=False)
            self.test_ds = HFProcessorFlowersDataset(raw_test, self.processor)
            print(f"Test: {len(self.test_ds)}")

    def train_dataloader(self):
        batch_size = self.config['data'].get('train_batch_size', self.config['data'].get('batch_size', 256))
        return DataLoader(
            self.train_ds, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=self.config['data'].get('num_workers', 4)
        )

    def val_dataloader(self):
        batch_size = self.config['data'].get('val_batch_size', self.config['data'].get('batch_size', 128))
        return DataLoader(
            self.val_ds, 
            batch_size=batch_size, 
            num_workers=self.config['data'].get('num_workers', 4)
        )

    def test_dataloader(self):
        batch_size = self.config['data'].get('batch_size', 128)
        return DataLoader(
            self.test_ds, 
            batch_size=batch_size, 
            num_workers=self.config['data'].get('num_workers', 4)
        )
