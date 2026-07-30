import json
import os
from datetime import datetime

RESULTS_JSONL = "/content/drive/MyDrive/ViT/results.jsonl"


def update_results(model, dataset, situation, accuracy, projector_type="", bridge_type=""):
    os.makedirs("outputs", exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_JSONL), exist_ok=True)

    records = []
    if os.path.exists(RESULTS_JSONL):
        with open(RESULTS_JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    ts = datetime.now().isoformat(timespec="seconds")
    for r in records:
        if (
            r.get("model") == model
            and r.get("dataset") == dataset
            and r.get("situation") == situation
            and r.get("projector_type") == projector_type
            and r.get("bridge_type") == bridge_type
        ):
            r["accuracy"] = accuracy
            r["updated_at"] = ts
            break
    else:
        records.append({
            "model":          model,
            "dataset":        dataset,
            "situation":      situation,
            "projector_type": projector_type,
            "bridge_type":    bridge_type,
            "accuracy":       accuracy,
            "updated_at":     ts,
        })

    with open(RESULTS_JSONL, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
