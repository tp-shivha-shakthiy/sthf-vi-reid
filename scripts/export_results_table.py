import csv
import json
import os


METRICS_PATH = "results/tables/main_comparison.csv"

COLUMNS = [
    "experiment",
    "ir_to_rgb_rank1",
    "ir_to_rgb_rank5",
    "ir_to_rgb_rank10",
    "ir_to_rgb_rank20",
    "ir_to_rgb_mAP",
    "rgb_to_ir_rank1",
    "rgb_to_ir_rank5",
    "rgb_to_ir_rank10",
    "rgb_to_ir_rank20",
    "rgb_to_ir_mAP",
]


def _flatten_metrics(experiment_name, metrics):
    row = {"experiment": experiment_name}
    for direction in ("ir_to_rgb", "rgb_to_ir"):
        for key, value in metrics.get(direction, {}).items():
            col = f"{direction}_{key}"
            row[col] = round(value, 2) if isinstance(value, float) else value
    return row


def append_experiment(experiment_name, metrics):
    """Append or update a row in the master comparison CSV.

    If the experiment already exists, its row is updated in-place
    (preserving other rows). Otherwise a new row is appended.

    Args:
        experiment_name: Short name for the experiment row.
        metrics: Dict with ``"ir_to_rgb"`` and ``"rgb_to_ir"`` sub-dicts,
            each containing metric keys (rank1, rank5, rank10, mAP, etc.).
    """
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    new_row = _flatten_metrics(experiment_name, metrics)
    rows = []

    if os.path.isfile(METRICS_PATH):
        with open(METRICS_PATH, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            updated = False
            for row in reader:
                if not updated and row.get("experiment") == experiment_name:
                    rows.append(new_row)
                    updated = True
                elif row.get("experiment") != experiment_name:
                    rows.append(row)
            if not updated:
                rows.append(new_row)
    else:
        fieldnames = COLUMNS
        rows.append(new_row)

    with open(METRICS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {experiment_name} in {METRICS_PATH}")


def load_metrics_from_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    import glob
    experiments_dir = "experiments"
    if not os.path.isdir(experiments_dir):
        print(f"No experiments directory found at {experiments_dir}")
        return

    for exp_dir in sorted(os.listdir(experiments_dir)):
        metrics_path = os.path.join(experiments_dir, exp_dir, "metrics.json")
        if not os.path.isfile(metrics_path):
            continue
        metrics = load_metrics_from_json(metrics_path)
        if "ir_to_rgb" in metrics and "rgb_to_ir" in metrics:
            append_experiment(exp_dir, metrics)
        else:
            print(f"Skipping {exp_dir}: missing cross-modality metrics")


if __name__ == "__main__":
    main()
