"""Simple stage logger for v4 preprocessing stages."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


class StageLogger:
    """Collect stage logs, mirror to UI sink, and persist to file."""

    def __init__(
        self,
        stage_name: str,
        run_name: str,
        log_path: Path,
        *,
        dry_run: bool = False,
        sink: Callable[[str], None] | None = None,
    ) -> None:
        self.stage_name = str(stage_name)
        self.run_name = str(run_name)
        self.log_path = log_path
        self.dry_run = bool(dry_run)
        self.sink = sink
        self._lines: list[str] = []

    def log(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp} UTC] {message}"
        self._lines.append(line)
        if self.sink is not None:
            self.sink(line)
        else:
            print(line)

    def log_parameters(self, parameters: dict[str, object]) -> None:
        self.log("Active parameters:")
        for key in sorted(parameters.keys()):
            self.log(f"  - {key}: {parameters[key]}")

    def log_summary(
        self,
        *,
        total_rows: int,
        successful_rows: int,
        failed_rows: int,
        skipped_rows: int,
        output_path: Path,
    ) -> None:
        self.log(f"Total rows: {int(total_rows)}")
        self.log(f"Successful rows: {int(successful_rows)}")
        self.log(f"Failed rows: {int(failed_rows)}")
        self.log(f"Skipped rows: {int(skipped_rows)}")
        self.log(f"Output path: {output_path}")

    def write(self) -> None:
        if self.dry_run:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("\n".join(self._lines) + "\n", encoding="utf-8")

    @property
    def lines(self) -> list[str]:
        return list(self._lines)
