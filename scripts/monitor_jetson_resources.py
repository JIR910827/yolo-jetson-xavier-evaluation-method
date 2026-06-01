#!/usr/bin/env python3
"""Record Jetson resource usage while an evaluation command is running.

This follows the older HYF performance-test workflow:
- jtop records CPU/GPU/RAM usage.
- iostat records disk I/O.
- ifstat records network traffic.

The script can either run for a fixed duration or wrap another command.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def require_command(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"Missing required command: {name}")
    return path


def open_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", newline="")


def terminate_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def ram_percent_from_meminfo() -> float | None:
    try:
        values: dict[str, int] = {}
        with Path("/proc/meminfo").open(encoding="utf-8") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0])
        total = values.get("MemTotal")
        available = values.get("MemAvailable", values.get("MemFree"))
        if not total or available is None:
            return None
        return round((total - available) / total * 100, 3)
    except OSError:
        return None


def normalize_jtop_stats(stats: dict[str, Any]) -> dict[str, Any]:
    row = dict(stats)
    if "time" in row:
        value = row["time"]
        row["time"] = value.strftime("%H:%M:%S") if isinstance(value, datetime) else str(value)
    cpu_values = [
        float(value)
        for key, value in row.items()
        if key.startswith("CPU") and isinstance(value, (int, float))
    ]
    row["CPU_Average"] = round(sum(cpu_values) / len(cpu_values), 3) if cpu_values else None
    row["RAM_Percent"] = ram_percent_from_meminfo()
    return row


def run_jtop_logger(output_csv: Path, stop_event: threading.Event) -> None:
    try:
        from jtop import jtop
    except ImportError as exc:
        raise SystemExit("Missing Python package: jetson-stats. Install it with: sudo -H pip3 install -U jetson-stats") from exc

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with jtop(interval=1) as jetson:
        first = normalize_jtop_stats(dict(jetson.stats))
        fieldnames = list(first.keys())
        for extra in ("CPU_Average", "RAM_Percent"):
            if extra not in fieldnames:
                fieldnames.append(extra)

        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(first)
            handle.flush()

            while not stop_event.is_set() and jetson.ok():
                time.sleep(1)
                writer.writerow(normalize_jtop_stats(dict(jetson.stats)))
                handle.flush()


def start_background_logs(args: argparse.Namespace) -> tuple[list[subprocess.Popen[Any]], list[Any], threading.Thread, threading.Event]:
    require_command("ifstat")
    require_command("iostat")

    handles = [
        open_log(args.output_dir / "Net" / "Net_log.txt"),
        open_log(args.output_dir / "Disk" / "Disk_log.txt"),
    ]

    processes = [
        subprocess.Popen(["ifstat", "-t", str(args.interval)], stdout=handles[0], stderr=subprocess.STDOUT),
        subprocess.Popen(["iostat", args.disk_device, "-dkt", str(args.interval)], stdout=handles[1], stderr=subprocess.STDOUT),
    ]

    stop_event = threading.Event()
    jtop_thread = threading.Thread(
        target=run_jtop_logger,
        args=(args.output_dir / "CPU_GPU_RAM" / "CPU_GPU_RAM_log.csv", stop_event),
        daemon=True,
    )
    jtop_thread.start()
    return processes, handles, jtop_thread, stop_event


def run_wrapped_command(command: list[str] | None, duration: int | None) -> int:
    if command:
        return subprocess.call(command)
    if duration is None:
        raise SystemExit("Use --duration seconds or pass a command after --")
    time.sleep(duration)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor Jetson CPU/GPU/RAM, disk I/O, and network traffic.")
    parser.add_argument("--output-dir", type=Path, default=Path("resource_monitor_results"))
    parser.add_argument("--duration", type=int, help="Monitor for this many seconds when no wrapped command is supplied.")
    parser.add_argument("--interval", type=int, default=1, help="Sampling interval in seconds.")
    parser.add_argument("--disk-device", default="mmcblk0", help="Disk device passed to iostat.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after -- while monitoring.")
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "monitor_command.txt").write_text(
        " ".join(command) if command else f"duration={args.duration}",
        encoding="utf-8",
    )

    processes: list[subprocess.Popen[Any]] = []
    handles: list[Any] = []
    stop_event: threading.Event | None = None
    jtop_thread: threading.Thread | None = None
    original_sigint = signal.getsignal(signal.SIGINT)
    exit_code = 1

    def request_stop(signum: int, frame: Any) -> None:
        if stop_event:
            stop_event.set()
        for process in processes:
            terminate_process(process)
        if callable(original_sigint):
            original_sigint(signum, frame)

    signal.signal(signal.SIGINT, request_stop)

    try:
        processes, handles, jtop_thread, stop_event = start_background_logs(args)
        exit_code = run_wrapped_command(command, args.duration)
    finally:
        if stop_event:
            stop_event.set()
        for process in processes:
            terminate_process(process)
        if jtop_thread:
            jtop_thread.join(timeout=10)
        for handle in handles:
            handle.close()

    print(f"Resource logs saved to: {args.output_dir}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
