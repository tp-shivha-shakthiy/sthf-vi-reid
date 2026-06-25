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


def _safe(val):
    if isinstance(val, (int, float)):
        return round(val, 2)
    return val if val is not None else ""


def _flatten_metrics(experiment_name, metrics):
    row = {col: "" for col in COLUMNS}
    row["experiment"] = experiment_name

    for direction in ("ir_to_rgb", "rgb_to_ir"):
        for key, value in metrics.get(direction, {}).items():
            col = f"{direction}_{key}"
            if col in row:
                row[col] = _safe(value)

    return row


def append_experiment(experiment_name, metrics):
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    new_row = _flatten_metrics(experiment_name, metrics)
    rows = []

    existing_fields = COLUMNS

    if os.path.isfile(METRICS_PATH):
        with open(METRICS_PATH, newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or COLUMNS

            for row in reader:
                if row.get("experiment") == experiment_name:
                    continue
                rows.append(row)

    rows.append(new_row)

    with open(METRICS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {experiment_name} in {METRICS_PATH}")


def load_metrics_from_json(path):
    with open(path) as f:
        return json.load(f)


def main():
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
            print(f"Skipping {exp_dir}: invalid metrics format")


if __name__ == "__main__":
    main()