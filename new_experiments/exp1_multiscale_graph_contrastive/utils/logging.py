"""
utils/logging.py
==================

Stage 1 reproducibility infrastructure: structured, per-experiment logging.

Per reproduction_protocol.md Section 14, the paper specifies no logging framework and no
intermediate-metric logging convention. The JSONL format and the specific metric set logged
here (`L_total` and its four components, plus per-session cumulative accuracy) are the
reproduction-engineering convention recorded in `configs/experiment.yaml`
(`logging_format`, `logged_metrics`), tagged IMPLEMENTATION_CHOICE -- never presented as a
paper fact.

This module only sets up the logging machinery (Stage 1). No training/evaluation code exists
yet to call it with real metrics; later stages will call `ExperimentLogger.log()`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class LoggingSetupError(Exception):
    pass


@dataclass
class LogRecord:
    """One structured JSONL log line."""

    timestamp_utc: str
    run_id: str
    event: str
    payload: Dict[str, Any] = field(default_factory=dict)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_python_logger(
    run_dir: str,
    name: str = "achg_clip",
    level: int = logging.INFO,
    also_console: bool = True,
) -> logging.Logger:
    """Configure and return a standard `logging.Logger` that writes to
    `{run_dir}/run.log` (plain text, human-readable) and, optionally, stdout.

    Idempotent: calling this twice for the same `name` will not duplicate handlers.
    """
    os.makedirs(run_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if setup is called more than once for the same logger name
    # (e.g. across multiple ExperimentLogger instances in tests).
    existing_files = {
        getattr(h, "baseFilename", None) for h in logger.handlers if isinstance(h, logging.FileHandler)
    }
    log_path = os.path.abspath(os.path.join(run_dir, "run.log"))
    if log_path not in existing_files:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)

    if also_console and not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(ch)

    return logger


class ExperimentLogger:
    """Writes structured JSONL records for one experiment run, plus a mirrored plain-text
    log via the standard `logging` module.

    File layout produced under `run_dir`:
        run.log        -- human-readable text log
        events.jsonl   -- one JSON object per line, one per `log()` call
    """

    def __init__(self, run_dir: str, run_id: str, logger_name: str = "achg_clip"):
        self.run_dir = run_dir
        self.run_id = run_id
        os.makedirs(run_dir, exist_ok=True)
        self.jsonl_path = os.path.join(run_dir, "events.jsonl")
        self.text_logger = setup_python_logger(run_dir, name=logger_name)

    def log(self, event: str, **payload: Any) -> LogRecord:
        """Write one structured event. `event` is a short string like 'epoch_end' or
        'session_eval'; `payload` is whatever key/value metrics belong to that event."""
        record = LogRecord(timestamp_utc=_utc_now_iso(), run_id=self.run_id, event=event, payload=payload)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")
        self.text_logger.info("event=%s payload=%s", event, payload)
        return record

    def log_loss_components(
        self,
        session: int,
        epoch: int,
        l_total: float,
        l_ce: float,
        l_recon: float,
        l_adv: float,
        l_energy: float,
    ) -> LogRecord:
        """Convenience wrapper matching `configs/experiment.yaml: logged_metrics` exactly
        (reproduction_protocol.md Section 14's reproduction-engineering convention)."""
        return self.log(
            "loss_components",
            session=session,
            epoch=epoch,
            L_total=l_total,
            L_CE=l_ce,
            L_recon=l_recon,
            L_adv=l_adv,
            L_energy=l_energy,
        )

    def log_session_accuracy(self, session: int, accuracy: float) -> LogRecord:
        return self.log("cumulative_session_accuracy", session=session, accuracy=accuracy)

    def read_events(self) -> list:
        """Read back all logged events (used by tests / later evaluation code)."""
        if not os.path.isfile(self.jsonl_path):
            return []
        records = []
        with open(self.jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


def make_run_id(dataset: str, timestamp: Optional[datetime] = None, git_hash: Optional[str] = None) -> str:
    """Build a run id following `configs/experiment.yaml: experiment_id_format`:
    `{dataset}_{YYYYMMDD}_{HHMMSS}_{short_git_hash}`.
    """
    ts = timestamp or datetime.now(timezone.utc)
    stamp = ts.strftime("%Y%m%d_%H%M%S")
    short_hash = (git_hash or "nogit")[:12]
    return f"{dataset}_{stamp}_{short_hash}"
