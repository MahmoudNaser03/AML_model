"""Promote a validated bundle locally; retain the previous directory for rollback."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from aml_app.service import AMLScorer


def promote(source, target, minimum_recall=.70, minimum_precision=.15):
    source, target = Path(source).resolve(), Path(target).resolve()
    if target == source or target in source.parents or source in target.parents:
        raise ValueError("Source and deployment directories must be separate.")
    if target.name != "production_artifacts":
        raise ValueError("Deployment target must be an explicit production_artifacts directory.")
    scorer = AMLScorer(source)
    report = json.loads((source / "evaluation.json").read_text())
    test = report["test"]
    if test["recall"] < minimum_recall or test["precision"] < minimum_precision:
        raise ValueError("Release does not meet the declared precision/recall acceptance gates.")
    staging = Path(tempfile.mkdtemp(prefix=".aml-release-", dir=target.parent))
    staging.chmod(0o755)  # The Docker service deliberately runs as a different non-root UID.
    for filename in ["manifest.json", *scorer.manifest["files"]]:
        shutil.copy2(source / filename, staging / filename)
    AMLScorer(staging)
    backup = None
    if target.exists():
        backup = target.with_name("production_artifacts.backup-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        target.rename(backup)
    try:
        staging.rename(target)
    except OSError:
        if backup: backup.rename(target)
        raise
    return {"model_version": scorer.version, "deployed": str(target), "rollback_directory": str(backup) if backup else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--target", type=Path, default=Path("production_artifacts"))
    parser.add_argument("--minimum-recall", type=float, default=.70)
    parser.add_argument("--minimum-precision", type=float, default=.15)
    args = parser.parse_args()
    print(json.dumps(promote(args.source, args.target, args.minimum_recall, args.minimum_precision), indent=2))


if __name__ == "__main__": main()
