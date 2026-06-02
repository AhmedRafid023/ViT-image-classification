import json
import os
from datetime import datetime

RESULTS_JSONL = "outputs/results.jsonl"


def update_results(model, dataset, situation, accuracy):
    os.makedirs("outputs", exist_ok=True)

    records = []
    if os.path.exists(RESULTS_JSONL):
        with open(RESULTS_JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    ts = datetime.now().isoformat(timespec="seconds")
    for r in records:
        if r["model"] == model and r["dataset"] == dataset and r["situation"] == situation:
            r["accuracy"] = accuracy
            r["updated_at"] = ts
            break
    else:
        records.append({
            "model":      model,
            "dataset":    dataset,
            "situation":  situation,
            "accuracy":   accuracy,
            "updated_at": ts,
        })

    with open(RESULTS_JSONL, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
