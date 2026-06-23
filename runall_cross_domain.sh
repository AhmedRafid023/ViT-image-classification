#!/bin/bash
# Train on ImageNet, then test on ImageNet and ImageNet variants.
# Skips a test combination if it already has an entry in outputs/results.jsonl.

TRAIN_DATASET=imagenet

TEST_DATASETS=(
    imagenet
    imagenet_sketch
    imagenetv2
)

MODELS=(
    # vit_b16
    vit_l16
    vit_l32_384
    # vit_h14
    # dinov2_b
    dinov2_l
)

SITUATIONS=(
    train_ve_out_ch
    train_proj_out_ch
    train_proj_ch
    train_ve_ch
    train_all
)

already_done() {
    python3 -c "
import json, sys
model, dataset, situation = sys.argv[1:]
try:
    with open('/content/drive/MyDrive/ViT/results.jsonl') as f:
        for line in f:
            r = json.loads(line.strip())
            if (
                r.get('model') == model
                and r.get('dataset') == dataset
                and r.get('situation') == situation
            ):
                sys.exit(0)
except FileNotFoundError:
    pass
sys.exit(1)
" "$1" "$2" "$3"
}

for MODEL in "${MODELS[@]}"; do
    for SIT in "${SITUATIONS[@]}"; do

        echo "=========================================="
        echo "Train | dataset=${TRAIN_DATASET} model=${MODEL} situation=${SIT}"
        echo "=========================================="

        bash scripts/run.sh train "${TRAIN_DATASET}" "${MODEL}" "${SIT}"

        CKPT=$(find outputs/wandb -name "${SIT}-epoch=*.ckpt" | xargs ls -t 2>/dev/null | head -1)

        if [ -z "$CKPT" ]; then
            echo "No checkpoint found for ${TRAIN_DATASET}/${MODEL}/${SIT} — skipping all tests"
            continue
        fi

        echo "Using checkpoint: ${CKPT}"

        for TEST_DATASET in "${TEST_DATASETS[@]}"; do

            if already_done "$MODEL" "$TEST_DATASET" "$SIT"; then
                echo "Skipping test ${TEST_DATASET}/${MODEL}/${SIT} — already in results.jsonl"
                continue
            fi

            echo "=========================================="
            echo "Test | train=${TRAIN_DATASET} test=${TEST_DATASET} model=${MODEL} situation=${SIT}"
            echo "=========================================="

            bash scripts/run.sh test "${TEST_DATASET}" "${MODEL}" "${SIT}" "${CKPT}"

        done

        echo "Deleting checkpoint: ${CKPT}"
        rm -f "${CKPT}"
        # also remove last.ckpt from the same run directory
        rm -f "$(dirname "${CKPT}")/last.ckpt"

    done
done
