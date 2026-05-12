"""CSV results writer. Streams one row per completed trial."""
import csv
import json
import os
from typing import IO

CSV_HEADER = [
    "algo", "maze_type", "pop_size", "instance_seed", "trial_seed",
    "success_overall", "success_postdisruption",
    "iterations", "path_length", "optimal_length", "path_optimality",
    "time_to_half_entropy", "diversity_floor", "mean_entropy", "adaptation_time",
    "entropy_history",
]


class ResultsWriter:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh: IO | None = None
        self._writer = None

    def __enter__(self):
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_HEADER)
        self._writer.writeheader()
        self._fh.flush()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fh:
            self._fh.flush()
            self._fh.close()

    def write(self, row: dict) -> None:
        serialized = dict(row)
        # JSON-encode entropy_history to keep CSV valid
        serialized["entropy_history"] = json.dumps(row.get("entropy_history", []))
        # None → empty string (CSV-friendly)
        for k in serialized:
            if serialized[k] is None:
                serialized[k] = ""
        self._writer.writerow(serialized)
        self._fh.flush()   # flush after each row — durable even on Ctrl-C
