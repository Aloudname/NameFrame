"""Project entry point.

Usage:
    python run.py
    python run.py --config config/config.yaml --set train.epochs=200
"""

from nameframe.run import run

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run NameFrame training")
    parser.add_argument("--config", "-c", type=str, default="config/config.yaml")
    parser.add_argument("--set", "-s", type=str, nargs="*", default=[])
    args = parser.parse_args()

    overrides: dict = {}
    for item in args.set:
        if "=" in item:
            k, v = item.split("=", 1)
            overrides[k] = v

    # import project modules to trigger registry registration
    import model.my_model   # noqa: F401
    import dataset.dataset  # noqa: F401

    result = run(config_path=args.config, config_overrides=overrides or None)
    print(f"Done. Best epoch: {result.best_epoch}")
