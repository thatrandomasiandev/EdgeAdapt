"""CLI smoke tests."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from edgeadapt.cli.main import main


def test_cli_help() -> None:
    """CLI entry points respond."""
    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0


def test_doctor() -> None:
    """doctor command prints JSON."""
    runner = CliRunner()
    r = runner.invoke(main, ["doctor"])
    assert r.exit_code == 0
    assert "edgeadapt_version" in r.output


def test_inspect_model(tmp_path: Path, dummy_model_paths: dict[str, object]) -> None:
    """inspect-model command prints IO metadata."""
    runner = CliRunner()
    p = dummy_model_paths["low"]
    r = runner.invoke(main, ["inspect-model", str(p)])
    assert r.exit_code == 0
    assert "inputs" in r.output


def test_validate(tmp_path: Path) -> None:
    """validate command runs."""
    p = tmp_path / "f.yaml"
    p.write_text(
        """
family: x
variants:
  a:
    path: dummy.onnx
    tier: high
    backend: onnx
    metadata:
      expected_latency_ms: 1
      memory_footprint_mb: 1
      accuracy_score: 0.5
      power_draw_estimate: low
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    r = runner.invoke(main, ["validate", "--config", str(p)])
    assert r.exit_code == 0


def test_benchmark_json(tmp_path: Path, dummy_model_paths: dict[str, object]) -> None:
    """benchmark --json emits BenchmarkReport v2 structure."""
    p = tmp_path / "f.yaml"
    p.write_text(
        f"""
family: bench
variants:
  a:
    path: {dummy_model_paths["low"]}
    tier: high
    backend: onnx
    metadata:
      expected_latency_ms: 1
      memory_footprint_mb: 1
      accuracy_score: 0.5
      power_draw_estimate: low
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    r = runner.invoke(main, ["benchmark", "--config", str(p), "--runs", "5", "--json"])
    assert r.exit_code == 0
    assert "latency_ms_p50" in r.output
    assert "metadata" in r.output
