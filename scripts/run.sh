#!/bin/bash
# Usage: bash scripts/run.sh <mode> <dataset> [model] [situation] [checkpoint] [extra hydra overrides...]
# Examples:
#   bash scripts/run.sh train fgvc_aircraft vit_l16 train_proj_ch
#   bash scripts/run.sh test  fgvc_aircraft vit_l16 train_proj_ch outputs/wandb/run-.../files/train_proj_ch-epoch=02.ckpt
#   bash scripts/run.sh train fgvc_aircraft vit_l16 train_proj_ch "" model.bridge_type=linear

MODE=$1
DATASET=$2
MODEL=${3:-"vit_b16"}
SITUATION=${4:-"train_all"}
CHECKPOINT=${5:-""}
shift 5
EXTRA="$@"

if [ "$MODE" == "test" ]; then
    python main.py mode=test dataset=${DATASET} model=${MODEL} experiment.situation=${SITUATION} "test.checkpoint='${CHECKPOINT}'" ${EXTRA}
else
    python main.py mode=train dataset=${DATASET} model=${MODEL} experiment.situation=${SITUATION} ${EXTRA}
fi
