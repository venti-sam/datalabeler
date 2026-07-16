"""datalabeler CLI - one subcommand per stage, all sharing config + manifest."""
from __future__ import annotations

import click

from .config import load_config
from .manifest import Manifest

_config_opt = click.option("-c", "--config", "config_path", default="config/pipeline.yaml",
                           show_default=True, help="pipeline config YAML")


@click.group()
def cli() -> None:
    """rosbag -> SAM 3 -> CVAT -> COCO auto-labeling pipeline."""


@cli.command()
@_config_opt
def extract(config_path: str) -> None:
    """Stage 1: rosbag -> sampled images + manifest."""
    from .stages.extract import extract as run
    cfg = load_config(config_path)
    stats = run(cfg)
    click.echo(f"extract: {stats}")


@cli.command()
@_config_opt
@click.option("--reannotate", is_flag=True, help="redo frames already auto-labeled")
def autolabel(config_path: str, reannotate: bool) -> None:
    """Stage 2: SAM 3 pre-annotations (run in the sam3 env)."""
    from .stages.autolabel import autolabel as run
    cfg = load_config(config_path)
    stats = run(cfg, reannotate=reannotate)
    click.echo(f"autolabel: {stats}")


@cli.command("cvat-export")
@_config_opt
@click.option("--bag", default=None, help="limit to one bag path")
@click.option("--batch", type=int, default=None, help="fixed frames-per-task instead of by-bag")
def cvat_export(config_path: str, bag: str | None, batch: int | None) -> None:
    """Stage 3: build CVAT-importable task folders (COCO)."""
    from .stages.cvat import export_for_cvat
    cfg = load_config(config_path)
    created = export_for_cvat(cfg, bag=bag, batch=batch)
    click.echo(f"cvat-export: {len(created)} task(s)")


@cli.command("cvat-import")
@_config_opt
@click.argument("export_json")
@click.option("--task-dir", default=None, help="task folder with manifest_map.json")
@click.option("--annotator", default=None, help="provenance: who corrected these")
def cvat_import(config_path: str, export_json: str, task_dir: str | None,
                annotator: str | None) -> None:
    """Stage 3: ingest CVAT COCO export -> canonical + status=corrected."""
    from .stages.cvat import import_from_cvat
    cfg = load_config(config_path)
    stats = import_from_cvat(cfg, export_json, task_dir=task_dir, annotator=annotator)
    click.echo(f"cvat-import: {stats}")


@cli.command("cvat-push")
@_config_opt
@click.option("--bag", default=None, help="limit to one bag path")
@click.option("--batch", type=int, default=None, help="fixed frames-per-task instead of by-bag")
def cvat_push(config_path: str, bag: str | None, batch: int | None) -> None:
    """Stage 3 (automated): create CVAT tasks + upload images & pre-annotations."""
    from .stages.cvat import cvat_push as run
    cfg = load_config(config_path)
    stats = run(cfg, bag=bag, batch=batch)
    click.echo(f"cvat-push: {stats}")


@cli.command("cvat-pull")
@_config_opt
@click.option("--name", default=None, help="one task name from the registry (default: all)")
@click.option("--annotator", default=None, help="provenance: who corrected these")
def cvat_pull(config_path: str, name: str | None, annotator: str | None) -> None:
    """Stage 3 (automated): export corrected COCO from CVAT -> canonical."""
    from .stages.cvat import cvat_pull as run
    cfg = load_config(config_path)
    stats = run(cfg, name=name, annotator=annotator)
    click.echo(f"cvat-pull: {stats}")


@cli.command()
@_config_opt
@click.option("--use-status", default="corrected", type=click.Choice(["corrected", "auto"]),
              help="package human-corrected labels, or raw auto labels for a dry run")
def package(config_path: str, use_status: str) -> None:
    """Stage 4: package (image, label) pairs + splits."""
    from .stages.package import package as run
    cfg = load_config(config_path)
    stats = run(cfg, use_status=use_status)
    click.echo(f"package: {stats}")


@cli.command()
@_config_opt
def status(config_path: str) -> None:
    """Show manifest counts by label status."""
    cfg = load_config(config_path)
    with Manifest(cfg.path("manifest")) as mani:
        click.echo(f"bags: {len(mani.bags())}")
        for st, n in sorted(mani.counts_by_status().items()):
            click.echo(f"  {st:10s} {n}")


if __name__ == "__main__":
    cli()
