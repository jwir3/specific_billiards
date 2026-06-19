import glob
import os
import time
from pathlib import Path


class Heartbeat:
    """Used by worker processes to report progress via files."""

    def __init__(self, directory=".progress"):
        self.dir = Path(directory)
        self.dir.mkdir(exist_ok=True)
        # Unique file per process/task
        self.file = self.dir / f"{os.getpid()}.progress"

    def report(self, current_tick):
        """Writes the current progress tick to the file."""
        with open(self.file, "w") as f:
            f.write(str(current_tick))

    def cleanup(self):
        """Removes the progress file when the task is done."""
        if self.file.exists():
            self.file.unlink()


class Monitor:
    """Used by the main process to aggregate progress."""

    def __init__(self, directory=".progress"):
        self.dir = Path(directory)

    def get_total_progress(self):
        """Sums up all values found in the progress directory."""
        total = 0
        for pfile in self.dir.glob("*.progress"):
            try:
                with open(pfile, "r") as f:
                    total += int(f.read().strip())
            except (ValueError, FileNotFoundError):
                continue
        return total

    def clear(self):
        """Wipes the progress directory."""
        for pfile in self.dir.glob("*.progress"):
            pfile.unlink()
