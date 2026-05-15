import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime
import mlflow

MLRUNS_DIR = os.path.join(os.getcwd(), "mlruns")
mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
mlflow.set_experiment("YuGen")


def _flatten(prefix: str, d: dict) -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(key, v))
        elif isinstance(v, (int, float)):
            out[key] = float(v)
    return out


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


class MLflowTracker:
    def __init__(self, run_name: str | None = None):
        self.run_name = run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run = None

    def __enter__(self):
        self.run = mlflow.start_run(run_name=self.run_name)
        mlflow.set_tag("git_sha", _git_sha())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run("FAILED" if exc_type else "FINISHED")
        return False

    def log_config(self, name: str, config) -> None:
        cfg = asdict(config) if is_dataclass(config) else dict(config)
        for k, v in cfg.items():
            if isinstance(v, (int, float, str, bool)):
                mlflow.log_param(f"{name}.{k}", v)

    def log_metrics_dict(self, prefix: str, metrics: dict) -> None:
        for k, v in _flatten(prefix, metrics).items():
            mlflow.log_metric(k, v)

    def log_json(self, local_path: str, artifact_subdir: str | None = None) -> None:
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path=artifact_subdir)

    def log_summary(self, split: str, summary_path: str) -> None:
        if not os.path.exists(summary_path):
            return
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        self.log_metrics_dict(split, summary)
        self.log_json(summary_path, artifact_subdir=f"evaluation/{split}")
