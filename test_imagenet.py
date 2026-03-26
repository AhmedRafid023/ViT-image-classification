import lightning as L
import yaml
import argparse
import json
import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from dataset_imagenet import TinyImageNetDataModule
from model_imagenet import ProjectorAblationModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    ckpt_path = config['test']['checkpoint_path']
    out_csv = config['test']['output_csv']
    print(f"Testing Checkpoint: {ckpt_path}")

    # 1. Load mappings
    with open(config['test']['mapping_file'], "r") as f:
        class_mapping = json.load(f)

    hf_dataset = load_dataset("zh-plus/tiny-imagenet", split="train")
    hf_label_names = hf_dataset.features['label'].names

    # 2. Setup Data
    dm = TinyImageNetDataModule(config)
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    # 3. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProjectorAblationModel.load_from_checkpoint(
        ckpt_path,
        config=config
    )
    model.to(device)
    model.eval()

    # 4. Evaluation Loop
    results = []
    all_preds = []
    all_labels = []

    print(f"\n--- Running Inference on {len(dm.test_ds)} Test Images ---")
    
    # FIX: Dynamically set the correct dtype for autocast based on your device
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    # FIX: Add torch.autocast to automatically bridge float32 inputs to float16 weights
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
        for batch in tqdm(test_loader):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            
            # Safely get paths (fallback to 'unknown' if not provided)
            paths = batch.get('image_path', ["unknown"] * len(labels))

            logits = model(pixel_values)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Store detailed results
            for i in range(len(labels)):
                true_idx = labels[i].item()
                pred_idx = preds[i].item()
                
                true_id = hf_label_names[true_idx]
                pred_id = hf_label_names[pred_idx]
                
                results.append({
                    "image_path": paths[i],
                    "true_label_idx": true_idx,
                    "pred_label_idx": pred_idx,
                    "true_label_name": class_mapping.get(true_id, "Unknown"),
                    "pred_label_name": class_mapping.get(pred_id, "Unknown"),
                    "is_correct": int(true_idx == pred_idx)
                })

    # 5. Save Results to CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Detailed results saved to: {out_csv}")

    # 6. Calculate and Print Accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = (all_preds == all_labels).mean()
    print("="*40)
    print(f"FINAL TEST ACCURACY: {accuracy * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()