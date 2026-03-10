#!/usr/bin/env python3
"""
Promote a registered model version to 'champion' (production-ready).

Usage:
    # Promote a specific version number:
    python scripts/promote_model.py --version 5

    # Promote the current 'challenger' (latest training run) to champion:
    python scripts/promote_model.py --promote-challenger

    # List all versions and their aliases:
    python scripts/promote_model.py --list

    # Override the MLflow tracking URI:
    python scripts/promote_model.py --version 5 --mlflow-uri http://127.0.0.1:8080
"""
import argparse
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "bank_churn_prediction"


def list_versions(client: MlflowClient) -> None:
    """Print all registered versions with their aliases and metrics."""
    try:
        registered_model = client.get_registered_model(MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        print(f"Model '{MODEL_NAME}' not found in the registry.")
        sys.exit(1)

    alias_map: dict[str, list[str]] = {}
    for alias in registered_model.aliases:
        version = client.get_model_version_by_alias(MODEL_NAME, alias).version
        alias_map.setdefault(version, []).append(alias)

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    versions = sorted(versions, key=lambda v: int(v.version), reverse=True)

    print(f"\n{'Ver':>4}  {'Aliases':<30}  {'Run ID':<34}  {'Created'}")
    print("-" * 110)
    for v in versions:
        aliases = ", ".join(alias_map.get(v.version, [])) or "-"
        created = v.creation_timestamp
        print(f"{v.version:>4}  {aliases:<30}  {v.run_id:<34}  {created}")

    print()


def promote(client: MlflowClient, version: str) -> None:
    """Set the 'champion' alias on the given version."""
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=version,
    )
    print(f"Version {version} is now the champion (production-ready).")
    print(f"All inference pipelines will load: models:/{MODEL_NAME}@champion")


def get_challenger_version(client: MlflowClient) -> str:
    """Return the version currently aliased as 'challenger'."""
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "challenger")
        return mv.version
    except mlflow.exceptions.MlflowException:
        print("No version with alias 'challenger' found. Run the training pipeline first.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a model version to 'champion' for production use.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--version", type=str,
        help="Model version number to promote to champion",
    )
    group.add_argument(
        "--promote-challenger", action="store_true",
        help="Promote the current 'challenger' version to champion",
    )
    group.add_argument(
        "--list", action="store_true",
        help="List all registered model versions and their aliases",
    )
    parser.add_argument(
        "--mlflow-uri", type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080"),
        help="MLflow tracking server URI",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    client = MlflowClient()

    if args.list:
        list_versions(client)
        return

    version = args.version
    if args.promote_challenger:
        version = get_challenger_version(client)
        print(f"Current challenger is version {version}.")

    promote(client, version)
    list_versions(client)


if __name__ == "__main__":
    main()
