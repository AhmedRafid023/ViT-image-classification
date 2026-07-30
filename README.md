# ViT-LLaVA Image Classification

A modular framework for ablating VLM projector components on standard image classification benchmarks. It plugs a frozen or trainable LLaVA projector between a ViT vision encoder and a linear classification head, and systematically studies which parts of the pipeline carry the most transferable representation.

---

## Project Structure

```
.
├── main.py                     # Single entry point (mode=train | mode=test)
├── runall.sh                   # Run all dataset/model/situation combinations
├── models/
│   └── vitllava.py             # ViT + LLaVA projector + classifier head
├── datasets/                   # One file per dataset (dassl-based)
│   └── datamodule.py           # Lightning DataModule wrapper
├── configs/
│   ├── config.yaml             # Root config
│   ├── dataset/                # Per-dataset configs (15 datasets)
│   ├── model/                  # Per-model configs (6 backbones)
│   ├── trainer/default.yaml
│   └── data/default.yaml
├── scripts/
│   └── run.sh                  # Train or test via Docker
├── utils/
│   └── tools.py                # Shared utilities (results JSONL writer)
├── download_data.sh            # Dataset download script
├── Dockerfile
└── .env.example
```

---

## Setup

**1. Create `.env` from the example and add your keys:**

```bash
cp .env.example .env
# fill in WANDB_API_KEY and HF_TOKEN
```

**2. Build the Docker image:**

```bash
docker build -t vit:latest .
```

---

## Data Preparation

Use `download_data.sh` to download any combination of datasets into a local `data/` directory. The data directory is mounted at `/data` inside the container.

**Download all datasets:**

```bash
docker run --gpus all --ipc=host --rm --env-file .env \
  -v ${PWD}:/app/ViT -v ${PWD}/data:/data \
  vit:latest bash download_data.sh --data /data --all
```

**Download specific datasets:**

```bash
docker run --gpus all --ipc=host --rm --env-file .env \
  -v ${PWD}:/app/ViT -v ${PWD}/data:/data \
  vit:latest bash download_data.sh --data /data --individual fgvc_aircraft,oxford_pets,food-101
```

**Download all except some:**

```bash
docker run --gpus all --ipc=host --rm --env-file .env \
  -v ${PWD}:/app/ViT -v ${PWD}/data:/data \
  vit:latest bash download_data.sh --data /data --all_except imagenet
```

---

## Usage

All commands are run inside the Docker container. The project directory is mounted at `/app/ViT` and the data directory at `/data`.

### Train

```bash
docker run --gpus all --ipc=host --rm --env-file .env \
  -v ${PWD}:/app/ViT -v ${PWD}/data:/data \
  vit:latest bash scripts/run.sh train <dataset> <model> <situation>
```

Example:

```bash
docker run --gpus all --ipc=host --rm --env-file .env \
  -v ${PWD}:/app/ViT -v ${PWD}/data:/data \
  vit:latest bash scripts/run.sh train fgvc_aircraft vit_l16 train_proj_ch
```

Test runs automatically after training by default (`test.after_train=true`). To disable:

```bash
docker run --gpus all --ipc=host --rm --env-file .env \
  -v ${PWD}:/app/ViT -v ${PWD}/data:/data \
  vit:latest bash scripts/run.sh train fgvc_aircraft vit_l16 train_proj_ch test.after_train=false
```

### Test (standalone)

```bash
docker run --gpus all --ipc=host --rm --env-file .env \
  -v ${PWD}:/app/ViT -v ${PWD}/data:/data \
  vit:latest bash scripts/run.sh test <dataset> <model> <situation> <checkpoint>
```

Example:

```bash
docker run --gpus all --ipc=host --rm --env-file .env \
  -v ${PWD}:/app/ViT -v ${PWD}/data:/data \
  vit:latest bash scripts/run.sh test fgvc_aircraft vit_l16 train_proj_ch \
  outputs/wandb/run-20260602_101320-aia2pxzi/files/train_proj_ch-epoch=02.ckpt
```

### Run All Combinations

```bash
docker run --gpus all --ipc=host --rm --env-file .env \
  -v ${PWD}:/app/ViT -v ${PWD}/data:/data \
  vit:latest bash runall.sh
```

Skips any combination already recorded in `outputs/results.jsonl`.

---

## Experiment Overview

Five training situations control which parts of the pipeline are frozen vs. trained:

### `train_ve_out_ch` — Bypass Projector
The vision encoder is fully frozen and the projector is bypassed entirely. Only the linear classification head is trained on raw VE features. This benchmarks the base encoder's off-the-shelf representation power.

### `train_proj_out_ch` — Frozen Projector
Both the VE and the projector are frozen. The head is trained on the projector's untuned output. Tests whether a pre-trained projector preserves or degrades visual features for classification.

### `train_proj_ch` — Train Projector + Head
The VE is frozen. The projector and classification head are trained together. The projector learns to translate frozen VE features into a space useful for the target classes.

### `train_ve_ch` — Train VE + Head
The projector is bypassed. The vision encoder and classification head are trained end-to-end, measuring how much the VE itself needs to adapt.

### `train_all` — Train Everything
VE, projector, and classification head are all trained together — the full-pipeline upper bound.

---

## Datasets

| Key | Dataset |
|---|---|
| `imagenet` | ImageNet-1K |
| `imagenet_a` | ImageNet-A |
| `imagenet_r` | ImageNet-R |
| `imagenetv2` | ImageNetV2 |
| `imagenet_sketch` | ImageNet-Sketch |
| `fgvc_aircraft` | FGVC Aircraft |
| `oxford_pets` | Oxford Pets |
| `oxford_flowers` | Oxford Flowers 102 |
| `food101` | Food-101 |
| `stanford_cars` | Stanford Cars |
| `caltech101` | Caltech-101 |
| `dtd` | DTD |
| `eurosat` | EuroSAT |
| `sun397` | SUN397 |
| `ucf101` | UCF-101 |

## Models

| Key | Backbone |
|---|---|
| `vit_b16` | ViT-B/16 (google/vit-base-patch16-224) |
| `vit_l16` | ViT-L/16 (google/vit-large-patch16-224) |
| `vit_l32_384` | ViT-L/32 @ 384px |
| `vit_h14` | ViT-H/14 |
| `dinov2_b` | DINOv2-B |
| `dinov2_l` | DINOv2-L |

## Situations

| Key | Description |
|---|---|
| `train_ve_out_ch` | Frozen VE, projector bypassed, train head only |
| `train_proj_out_ch` | Frozen VE + projector, train head only |
| `train_proj_ch` | Frozen VE, train projector + head |
| `train_ve_ch` | Train VE + head, projector bypassed |
| `train_all` | Train VE + projector + head |

---

## Outputs

All outputs for a run are saved inside the wandb run directory:

```
outputs/
  wandb/
    run-<date>-<id>/
      files/
        config.yaml                    # resolved Hydra config
        <situation>-epoch=XX.ckpt      # best checkpoint
        last.ckpt
        <Dataset>_results.csv          # per-image predictions
  results.jsonl                        # accuracy summary across all runs
```
