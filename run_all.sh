#!/bin/bash
# Run train then test for every dataset/model/situation/projector/bridge combination.
# Skips a combination if it already has an entry in outputs/results.jsonl.

DATASETS=(
    # imagenet
    oxford_pets
    oxford_flowers
    dtd
    food101
    eurosat
    caltech101
    fgvc_aircraft
    stanford_cars
    sun397
    ucf101
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
    # train_ve_out_ch
    # train_proj_out_ch
    train_proj_ch
    # train_ve_ch
    # train_all
)

PROJECTOR_TYPES=(
    llava
    blip2
    paligemma
)

BRIDGE_TYPES=(
    linear
    mlp
)

already_done() {
    python3 -c "
import json, sys
model, dataset, situation = sys.argv[1:]
try:
    with open('outputs/results.jsonl') as f:
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
    for DATASET in "${DATASETS[@]}"; do
        for SIT in "${SITUATIONS[@]}"; do
            for PROJ in "${PROJECTOR_TYPES[@]}"; do
                for BRIDGE in "${BRIDGE_TYPES[@]}"; do

                    if already_done "$MODEL" "$DATASET" "$SIT"; then
                        echo "Skipping ${DATASET}/${MODEL}/${SIT}/${PROJ}/${BRIDGE} — already in results.jsonl"
                        continue
                    fi

                    echo "=========================================="
                    echo "Train | dataset=${DATASET} model=${MODEL} situation=${SIT} proj=${PROJ} bridge=${BRIDGE}"
                    echo "=========================================="
                    bash scripts/run.sh train "${DATASET}" "${MODEL}" "${SIT}" "" "model.projector_type=${PROJ}" "model.bridge_type=${BRIDGE}" "test.after_train=true"

                done
            done
        done
    done
done