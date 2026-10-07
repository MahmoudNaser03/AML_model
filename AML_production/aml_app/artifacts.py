"""Validate a complete artifact release before making the scorer ready."""
import hashlib
import json
from pathlib import Path

from .entity_features import ENTITY_FEATURE_VERSION
from .entity_features import source_hash as entity_source_hash
from .pipeline import FEATURE_VERSION, source_hash


class ArtifactError(ValueError):
    pass


def validate_bundle(directory, *, allow_development=False):
    root = Path(directory)
    try:
        manifest = json.loads((root / "manifest.json").read_text())
        schema = manifest["schema_version"]
        if schema not in {2, 3} or manifest["feature_version"] != FEATURE_VERSION:
            raise ArtifactError("Artifact feature version is incompatible; retrain/export using the shared pipeline.")
        if manifest["feature_source_sha256"] != source_hash():
            raise ArtifactError("Feature implementation differs from the trained release.")
        if manifest.get("development_only", True) and not allow_development:
            raise ArtifactError("Development-only artifacts cannot be served in production.")
        required = {"model_A_general_aml.json", "model_B_pattern_specialist.json", "feature_columns.csv",
                    "thresholds.json", "decision_policy.json", "reference.json", "evaluation.json", "splits.json"}
        if schema == 3:
            if manifest.get("entity_feature_version") != ENTITY_FEATURE_VERSION:
                raise ArtifactError("Entity feature version is incompatible.")
            if manifest.get("entity_feature_source_sha256") != entity_source_hash():
                raise ArtifactError("Entity feature implementation differs from the trained release.")
            required |= {"base_feature_columns.csv", "entity_registry.npz", "model_baseline_b.json"}
        if not required.issubset(manifest["files"]):
            raise ArtifactError("Incomplete manifest.")
        for filename, expected in manifest["files"].items():
            if Path(filename).name != filename:
                raise ArtifactError("Invalid artifact filename.")
            if hashlib.sha256((root / filename).read_bytes()).hexdigest() != expected:
                raise ArtifactError("Artifact checksum mismatch: " + filename)
        return manifest
    except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError) as exc:
        raise ArtifactError("Artifact bundle is incomplete. Export a validated version-2 release.") from exc
