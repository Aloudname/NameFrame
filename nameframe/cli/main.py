"""Command-line interface for NameFrame project management and training."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main CLI entry point.

    Args:
        argv: Command-line arguments. Uses ``sys.argv`` if ``None``.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="nameframe",
        description="NameFrame — standardized deep learning training framework",
    )
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser] = (
        parser.add_subparsers(dest="command", help="Available commands")
    )

    # init
    init_parser: argparse.ArgumentParser = subparsers.add_parser(
        "init", help="Initialize a new NameFrame project"
    )
    init_parser.add_argument(
        "project_name", type=str, help="Name of the new project directory"
    )
    init_parser.add_argument(
        "--template", type=str, default=None,
        help="Git URL or local path to a custom template (default: built-in)",
    )

    # run
    run_parser: argparse.ArgumentParser = subparsers.add_parser(
        "run", help="Run training in the current project"
    )
    run_parser.add_argument(
        "--config", "-c", type=str, default="config/config.yaml",
        help="Path to the YAML config file"
    )
    run_parser.add_argument(
        "--set", "-s", type=str, nargs="*", default=[],
        help="Override config values: --set train.lr=0.001 train.epochs=200"
    )

    # list
    list_parser: argparse.ArgumentParser = subparsers.add_parser(
        "list", help="List available registered components"
    )
    list_parser.add_argument(
        "component", type=str, nargs="?", default="all",
        choices=["models", "datasets", "losses", "metrics", "all"],
        help="Component type to list"
    )

    # ops
    ops_parser: argparse.ArgumentParser = subparsers.add_parser(
        "ops", help="Manage accelerated ops"
    )
    ops_sub: argparse._SubParsersAction[argparse.ArgumentParser] = (
        ops_parser.add_subparsers(dest="ops_command", help="Ops subcommand")
    )
    ops_sub.add_parser("build", help="Pre-build all registered native ops")
    ops_sub.add_parser("status", help="Show ops status matrix")
    ops_verify: argparse.ArgumentParser = ops_sub.add_parser("verify", help="Verify native ops correctness")
    ops_verify.add_argument("--tolerance", type=float, default=1e-4, help="Allowed error")
    ops_sub.add_parser("clean", help="Clear compilation cache")

    args: argparse.Namespace = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    if args.command == "init":
        _cmd_init(args.project_name, args.template)
    elif args.command == "run":
        _cmd_run(args.config, args.set)
    elif args.command == "list":
        _cmd_list(args.component)
    elif args.command == "ops":
        _cmd_ops(args)


def _cmd_init(project_name: str, template: Optional[str]) -> None:
    """Scaffold a new project from the built-in or custom template.

    Args:
        project_name: Target directory name.
        template: Optional path or URL to a custom template.
    """
    target: Path = Path(project_name).resolve()
    if target.exists():
        print(f"Error: '{target}' already exists.", file=sys.stderr)
        sys.exit(1)

    if template is not None:
        _init_from_template(target, template)
    else:
        _init_builtin(target)

    print(f"\nProject '{project_name}' created at {target}")
    print("Next steps:")
    print(f"  cd {project_name}")
    print("  pip install -e ../nameframe  # or install nameframe package")
    print("  nameframe run")


def _init_builtin(target: Path) -> None:
    """Copy the built-in template directory to *target*."""
    template_src: Path = Path(__file__).resolve().parent.parent.parent / "template"
    if not template_src.exists():
        print("Error: built-in template not found.", file=sys.stderr)
        sys.exit(1)
    shutil.copytree(str(template_src), str(target))


def _init_from_template(target: Path, template_url: str) -> None:
    """Clone a template from a git URL or copy from a local path.

    Args:
        target: Destination directory.
        template_url: Git URL or local path.
    """
    if os.path.exists(template_url):
        shutil.copytree(template_url, str(target))
    else:
        import subprocess
        subprocess.run(
            ["git", "clone", template_url, str(target)],
            check=True,
        )
        # remove .git so the new project starts fresh
        git_dir: Path = target / ".git"
        if git_dir.exists():
            shutil.rmtree(str(git_dir))


def _cmd_run(config_path: str, overrides: List[str]) -> None:
    """Run training from the current project directory.

    Args:
        config_path: Path to YAML config.
        overrides: List of ``key=value`` strings.
    """
    from nameframe.config import load_config, merge_args

    cfg_path: Path = Path(config_path)
    if not cfg_path.exists():
        # try project-level config
        cfg_path = Path("config") / config_path
    if not cfg_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config: Any = load_config(str(cfg_path))

    if overrides:
        parsed: Dict[str, Any] = {}
        for item in overrides:
            if "=" not in item:
                print(f"Warning: skipping invalid override '{item}' (expected key=value)")
                continue
            key, value = item.split("=", 1)
            parsed[key] = _parse_value(value)
        config = merge_args(config, parsed)

    from nameframe.run import run as pipeline_run
    result: Any = pipeline_run(config)
    print(f"\nTraining complete. Best epoch: {result.best_epoch}, "
          f"metric: {result.best_val_metric:.4f}")


def _cmd_list(component: str) -> None:
    """Print registered components.

    Args:
        component: Component type or ``"all"``.
    """
    from nameframe.registry import (
        DATASET_REGISTRY,
        LOSS_REGISTRY,
        METRIC_REGISTRY,
        MODEL_REGISTRY,
    )

    registries: Dict[str, Any] = {
        "models": MODEL_REGISTRY,
        "datasets": DATASET_REGISTRY,
        "losses": LOSS_REGISTRY,
        "metrics": METRIC_REGISTRY,
    }

    targets: Dict[str, Any] = (
        registries if component == "all" else {component: registries.get(component)}
    )

    for cat, reg in targets.items():
        if reg is None:
            continue
        names: List[str] = reg.list()
        print(f"\n[{cat}] ({len(names)} registered):")
        for name in names:
            print(f"  - {name}")


def _cmd_ops(args: argparse.Namespace) -> None:
    """Execute an ops subcommand.

    Args:
        args: Parsed CLI arguments (expects ``ops_command`` attribute).
    """
    cmd: str = getattr(args, "ops_command", None) or "status"

    if cmd == "status":
        from nameframe.ops import status as ops_status
        result: Dict[str, Dict[str, Any]] = ops_status()
        if not result:
            print("No ops registered.")
            return
        print(f"{'OP NAME':<30} {'BACKEND':<12} {'BUILT':<8}")
        print("-" * 50)
        for op_name, info in result.items():
            built: str = "yes" if info["built"] else "no"
            print(f"{op_name:<30} {info['backend']:<12} {built:<8}")

    elif cmd == "build":
        from nameframe.ops import build_all as ops_build
        from nameframe.config import load_config
        # try to load config for ops settings
        try:
            config: Any = load_config("config/config.yaml")
            ops_config: Any = getattr(config, "ops", {})
        except Exception:
            ops_config = {}
        status_map: Dict[str, str] = ops_build(ops_config)
        print(f"Built {len(status_map)} ops.")
        for op_name, backend in status_map.items():
            print(f"  {op_name} -> {backend}")

    elif cmd == "verify":
        tolerance: float = getattr(args, "tolerance", 1e-4)
        from nameframe.ops.verify import verify_all
        results: Dict[str, bool] = verify_all(tolerance)
        passed: int = sum(1 for v in results.values() if v)
        print(f"Verified {passed}/{len(results)} ops passed.")
        for key, ok in results.items():
            print(f"  {key}: {'PASS' if ok else 'FAIL'}")

    elif cmd == "clean":
        import shutil
        cache_dir: str = os.path.expanduser("~/.cache/nameframe/ops")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Cleared {cache_dir}")


def _parse_value(raw: str) -> Any:
    """Parse a CLI override value string into the appropriate Python type.

    Args:
        raw: Raw string value (e.g. ``"0.001"``, ``"true"``, ``"200"``).

    Returns:
        Parsed value: ``float``, ``int``, ``bool``, or ``str``.
    """
    low: str = raw.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


if __name__ == "__main__":
    main()
