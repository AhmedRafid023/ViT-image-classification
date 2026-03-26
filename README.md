

```markdown
# VLM Projector Ablation & Fine-Tuning on Tiny ImageNet

This repository contains a modular framework for evaluating, ablating, and fine-tuning Vision-Language Model (VLM) components on the 200-class **Tiny ImageNet** dataset. We systematically isolate the Vision Encoder (e.g., CLIP, ViT) and the Multimodal Projector (LLaVA/Gemma styles) using both full-parameter freezing and **Low-Rank Adaptation (LoRA)** to determine where feature alignment happens best.

> **Note:** Hugging Face tokens and other environmental variables are securely managed via a `.env` file.

---

## 📂 Project Structure

```text
.
├── configs/
│   ├── train.yaml          # Configuration for ablation situations, models, and hyperparameters
│   └── test.yaml           # Configuration for checkpoint evaluation
├── data/
│   ├── validation/         # Physical 9K validation images (generated locally)
│   ├── test/               # Physical 1K test images (generated locally)
│   ├── validation.csv      # Paths and labels for the validation split
│   └── test.csv            # Paths and labels for the test split
├── class_mapping.json      # Cleaned WordNet ID to human-readable string mapping
├── dataset_imagenet.py     # DataModule handling HF streaming and local CSV loading
├── model_imagenet.py       # LightningModule with dynamic Projector & LoRA PEFT routing
├── process_imagenet.py     # Data pipeline: downloading mapping, splitting, and saving images
├── train_imagenet.py       # Main training script
├── test_imagenet.py        # Checkpoint testing and visual validation script
├── Dockerfile              # Environment with PyTorch, PEFT, and CUDA dependencies
├── .env                    # Environment variables (HF_TOKEN, etc.)
└── README.md               # Project documentation
```

---

## 🧬 Experiment Overview



This research explores how visual features propagate through different VLM routing strategies and how LoRA adapters impact representation learning. We test 5 distinct situations controlled via the `train.yaml` config:

### 1. Situation: `train_ve_out_ch` (Bypass Projector)
The Vision Encoder (VE) is completely frozen. The projector is **bypassed entirely**. Only a linear classification head (CH) is trained on the raw VE features. This benchmarks the raw representation power of the base encoder (e.g., CLIP).

### 2. Situation: `train_proj_out_ch` (Frozen Projector)
Both the VE and the Projector are frozen. Only the CH is trained on the projector's output. This tests if an untrained/pre-trained projector degrades or preserves visual features.

### 3. Situation: `train_proj_ch` (Train Projector)
The VE is frozen. Both the Projector and the CH are unfozen and actively learning. This trains a specialized "translation" layer for the 200 Tiny ImageNet classes without altering the base vision model.

### 4. Situation: `train_ve_ch` (Train Vision Encoder)
The projector is bypassed. We apply **LoRA** (Low-Rank Adaptation) exclusively to the attention matrices (`q_proj`, `v_proj`) of the Vision Encoder, training the adapters alongside the CH.

### 5. Situation: `train_all` (Full Finetune Alignment)
LoRA adapters are applied to **both** the Vision Encoder and the Projector. The adapters and the CH are trained simultaneously, representing a highly parameter-efficient full-pipeline fine-tune.

---

## 🛠️ Execution Guide

### 1. Environment Setup

Build the Docker image to ensure all dependencies (`transformers`, `peft`, `lightning`) and CUDA drivers are configured. Ensure your `.env` file is populated with your `HF_TOKEN`.

```bash
docker build -t vlm-ablation-exp .

# Run container with GPU access and pass the environment variables
docker run --gpus all -it --rm -v $(pwd):/app --env-file .env vlm-ablation-exp /bin/bash
```

### 2. Data Preparation

Prepare the Tiny ImageNet dataset. This script downloads the class mapping from Hugging Face, cleans the WordNet IDs into human-readable strings, splits the 10K validation set into 9K Val / 1K Test, and saves the physical images and CSVs to disk for reproducible testing.

```bash
python process_imagenet.py
```

### 3. Running Experiments

Configure your desired model (`openai/clip-vit-base-patch32` vs `google/vit-base-patch32-384`), projector type (`llava`, `gemma`, `none`), and experiment `situation` inside `configs/train.yaml`.

**Start Training:**

```bash
docker run --gpus all --ipc=host --rm --env-file .env -v $(pwd):/app vlm-ablation-exp \
    python train_imagenet.py --config configs/train.yaml
```

### 4. Evaluation

Update `configs/test.yaml` with the path to your best generated `.ckpt` file. This will evaluate the checkpoint on the static 1K test split and print a visual sanity check mapping the integer predictions back to human-readable class names.

```bash
docker run --gpus all --ipc=host --rm --env-file .env -v $(pwd):/app vlm-ablation-exp \
    python test_imagenet.py --config configs/test.yaml
```
```# ViT-image-classification
