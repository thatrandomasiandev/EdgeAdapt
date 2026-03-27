"""Click CLI for EdgeAdapt."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

from edgeadapt.benchmark.runner import benchmark_family
from edgeadapt.profiler.base import get_profiler
from edgeadapt.registry.family import ModelFamily
from edgeadapt.tooling.environment import environment_report
from edgeadapt.tooling.onnx_inspect import describe_onnx_model


@click.group()
def main() -> None:
    """EdgeAdapt command-line interface."""


@main.command("benchmark")
@click.option(
    "--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option("--sample", "sample_path", type=click.Path(exists=True, path_type=Path))
@click.option("--runs", default=100, show_default=True, type=int)
def benchmark_cmd(config_path: Path, sample_path: Path | None, runs: int) -> None:
    """Benchmark all variants from a family YAML and print a table."""
    family = ModelFamily.from_yaml(str(config_path))
    resolved = family.resolve_paths(config_path.parent)
    sample = np.load(sample_path) if sample_path is not None else None
    report = benchmark_family(resolved, sample, runs=runs)
    click.echo("variant\tlatency_ms\tmemory_mb")
    for name, row in report.rows.items():
        click.echo(f"{name}\t{row['latency_ms']:.4f}\t{row['memory_mb']:.4f}")


@main.command("profile")
@click.option("--interval", default=1.0, show_default=True, type=float)
@click.option("--duration", default=30.0, show_default=True, type=float)
def profile_cmd(interval: float, duration: float) -> None:
    """Stream DeviceState readings as JSON lines."""
    import time  # noqa: PLC0415

    profiler = get_profiler()
    end = time.monotonic() + duration
    while time.monotonic() < end:
        state = profiler.read()
        payload = {
            "battery_pct": state.battery_pct,
            "is_charging": state.is_charging,
            "thermal_zone_c": state.thermal_zone_c,
            "available_ram_mb": state.available_ram_mb,
            "cpu_utilization_pct": state.cpu_utilization_pct,
            "gpu_utilization_pct": state.gpu_utilization_pct,
            "network_quality": state.network_quality,
            "timestamp": state.timestamp,
        }
        click.echo(json.dumps(payload))
        time.sleep(interval)


@main.command("validate")
@click.option(
    "--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True
)
def validate_cmd(config_path: Path) -> None:
    """Validate a family YAML and check that model paths exist."""
    family = ModelFamily.from_yaml(str(config_path))
    resolved = family.resolve_paths(config_path.parent)
    click.echo(f"family={resolved.name} variants={len(resolved.variants)}")
    for name, v in resolved.variants.items():
        exists = Path(v.path).is_file()
        click.echo(f"  {name}: path={v.path} exists={exists}")


@main.command("doctor")
def doctor_cmd() -> None:
    """Print environment and dependency diagnostics as JSON."""
    click.echo(json.dumps(environment_report(), indent=2))


@main.command("inspect-model")
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
def inspect_model_cmd(model_path: Path) -> None:
    """Print ONNX Runtime IO metadata for a single ``.onnx`` file."""
    click.echo(json.dumps(describe_onnx_model(model_path), indent=2))


if __name__ == "__main__":
    main()
