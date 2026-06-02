#!/bin/bash
# Run train (+ optional test) for every dataset/model/situation combination.
# Skips a combination if it already has an entry in outputs/results.jsonl.
# If test.after_train=false in config, runs test separately after train.

DATASETS=(imagenet oxford_pets oxford_flowers dtd food101 eurosat caltech101 fgvc_aircraft stanford_cars sun397 ucf101)
MODELS=(vit_b16 vit_l16 vit_l32_384 vit_h14 dinov2_b dinov2_l)
SITUATIONS=(train_ve_out_ch train_proj_out_ch train_proj_ch train_ve_ch train_all)

TEST_AFTER_TRAIN=$(python3 -c "
import yaml
with open('configs/config.yaml') as f:
    cfg = yaml.safe_load(f)
print(str(cfg.get('test', {}).get('after_train', True)).lower())
")

already_done() {
    python3 -c "
import json, sys
model, dataset, situation = sys.argv[1:]
try:
    with open('outputs/results.jsonl') as f:
        for line in f:
            r = json.loads(line.strip())
            if r.get('model')==model and r.get('dataset')==dataset and r.get('situation')==situation:
                sys.exit(0)
except FileNotFoundError:
    pass
sys.exit(1)
" "$1" "$2" "$3"
}

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for SIT in "${SITUATIONS[@]}"; do

            if already_done "$MODEL" "$DATASET" "$SIT"; then
                echo "Skipping ${DATASET}/${MODEL}/${SIT} — already in results.jsonl"
                continue
            fi

            echo "=========================================="
            echo "Train | dataset=${DATASET} model=${MODEL} situation=${SIT}"
            echo "=========================================="
            bash scripts/run.sh train ${DATASET} ${MODEL} ${SIT}

            if [ "$TEST_AFTER_TRAIN" == "false" ]; then
                CKPT=$(find outputs/wandb -name "${SIT}-epoch=*.ckpt" | xargs ls -t 2>/dev/null | head -1)
                if [ -n "$CKPT" ]; then
                    echo "=========================================="
                    echo "Test  | dataset=${DATASET} model=${MODEL} situation=${SIT}"
                    echo "=========================================="
                    bash scripts/run.sh test ${DATASET} ${MODEL} ${SIT} "${CKPT}"
                else
                    echo "No checkpoint found for ${DATASET}/${MODEL}/${SIT} — skipping test"
                fi
            fi

        done
    done
done
