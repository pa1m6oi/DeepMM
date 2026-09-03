from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import torch


REQUIRED_METADATA = (
    "model_kind", "Nt", "Nr", "Ne", "snr_db", "layer_count",
    "channel_law", "ideal_csi", "port_selection", "fluid_antennas",
)

STRICT_TRAINING_METADATA = (
    "training_samples", "epochs", "batch_size", "train_size", "learning_rate",
    "reference_update_max_iter", "supervisor_epoch_num", "default_inference_layers",
)
SUPPORTED_DEEP_MM_LAYER_COUNTS = tuple(range(1, 17))


def build_metadata(model_kind: str, Nt: int, Nr: int, Ne: int, snr_db: float, layer_count: int, **extra: Any) -> Dict[str, Any]:
    data = {
        "model_kind": str(model_kind), "Nt": int(Nt), "Nr": int(Nr), "Ne": int(Ne),
        "snr_db": float(snr_db), "layer_count": int(layer_count),
        "channel_law": "iid_complex_gaussian_unit_power", "ideal_csi": True,
        "port_selection": False, "fluid_antennas": False,
    }
    data.update(extra)
    return data


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    metadata: Dict[str, Any],
    optimizer=None,
    epoch: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
    strict: bool = True,
) -> str:
    missing = [key for key in REQUIRED_METADATA + STRICT_TRAINING_METADATA if key not in metadata]
    if missing:
        raise ValueError("checkpoint metadata missing: " + ", ".join(missing))
    is_small_sample = metadata.get("artifact_scope") in {"small_sample", "early_stopped"}
    expected = {
        "training_samples": 50000, "epochs": 100, "batch_size": 200,
        "train_size": 0.8, "learning_rate": 5e-3, "reference_update_max_iter": 100,
        "default_inference_layers": 6,
    }
    mismatches = []
    if strict and not is_small_sample:
        for key, value in expected.items():
            actual = metadata.get(key)
            valid = abs(float(actual) - value) <= 1e-9 if isinstance(value, float) else actual == value
            if not valid:
                mismatches.append("{}={} (expected {})".format(key, actual, value))
        supervisor_expected = 0.20 if metadata.get("model_kind") == "deep_mm" else 0.0
        if abs(float(metadata.get("supervisor_epoch_num")) - supervisor_expected) > 1e-9:
            mismatches.append("supervisor_epoch_num={} (expected {})".format(metadata.get("supervisor_epoch_num"), supervisor_expected))
    if strict and not is_small_sample:
        if metadata.get("model_kind") == "deep_mm" and metadata.get("layer_count") not in SUPPORTED_DEEP_MM_LAYER_COUNTS:
            mismatches.append("layer_count={} (expected 1 through 16)".format(metadata.get("layer_count")))
        if metadata.get("model_kind") == "iaidnn" and metadata.get("layer_count") != 16:
            mismatches.append("layer_count={} (expected 16)".format(metadata.get("layer_count")))
        if metadata.get("model_kind") == "gnn" and metadata.get("layer_count") != 1:
            mismatches.append("layer_count={} (expected 1)".format(metadata.get("layer_count")))
        if (
            metadata.get("validation_enabled") is False
            and int(epoch) != int(metadata.get("epochs", epoch))
        ):
            mismatches.append(
                "epoch={} (expected final epoch {})".format(
                    epoch, metadata.get("epochs")
                )
            )
    if mismatches:
        raise ValueError("checkpoint strict training configuration mismatch: " + "; ".join(mismatches))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1, "metadata": dict(metadata),
        "model_config": {
            "class": model.__class__.__name__, "layers": int(getattr(model, "layers", metadata["layer_count"])),
            "Nt": int(getattr(model, "Nt", metadata["Nt"])),
            "Nr": int(getattr(model, "Nr", metadata["Nr"])),
            "Ne": int(getattr(model, "Ne", metadata["Ne"])),
            "Nd": int(getattr(model, "Nd", metadata.get("Nd", metadata["Nt"]))),
            "hidden_size": int(getattr(model, "hidden_size", 0)), "embed_dim": int(getattr(model, "embed_dim", 0)),
            "message_passing_steps": int(getattr(model, "message_passing_steps", 0)), "factor_rank": int(getattr(model, "factor_rank", 0)),
            "learn_mode": str(getattr(model, "learn_mode", "direct")),
            "scale_min": float(getattr(model, "scale_min", 0.25)),
            "scale_max": float(getattr(model, "scale_max", 4.0)),
            "nt_norm": float(getattr(model, "nt_norm", 16.0)),
            "inverse_steps": int(getattr(model, "inverse_steps", 0)),
            "model_arch": str(getattr(model, "model_arch", metadata.get("model_arch", ""))),
            "rgnn_layers": int(getattr(model, "rgnn_layers", metadata.get("rgnn_layers", 0))),
            "architecture": str(getattr(model, "architecture", metadata.get("architecture", "legacy_wmmse_xyzo"))),
            "architecture_version": int(getattr(model, "architecture_version", metadata.get("architecture_version", 1))),
            "formal_iaidnn": bool(getattr(model, "formal_iaidnn", metadata.get("architecture") == "normalized_wiretap")),
            "paper_iaidnn": bool(getattr(model, "paper_iaidnn", metadata.get("architecture") == "paper_wiretap")),
            "inference_modes": list(metadata.get("inference_modes", ["legacy"])),
            "monotone_safeguard": bool(getattr(model, "monotone_safeguard", metadata.get("monotone_safeguard", False))),
            "monotone_acceptance": bool(getattr(model, "monotone_acceptance", metadata.get("monotone_acceptance", False))),
            "backtracking_steps": int(getattr(model, "backtracking_steps", metadata.get("backtracking_steps", 0))),
            "acceptance_tol": float(getattr(model, "acceptance_tol", metadata.get("acceptance_tol", 1e-6))),
            "initialization": str(getattr(model, "initialization", metadata.get("initialization", "isotropic"))),
            "learned_blocks": (
                None if getattr(model, "learned_blocks", None) is None
                else list(getattr(model, "learned_blocks"))
            ),
        },
        "state_dict": model.state_dict(), "epoch": int(epoch), "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, str(tmp))
    tmp.replace(target)
    return sha256_file(str(target))


def load_checkpoint(path: str, model: torch.nn.Module, expected: Optional[Dict[str, Any]] = None, device: str = "cpu") -> Dict[str, Any]:
    payload = torch.load(path, map_location=device)
    metadata = payload.get("metadata", {})
    model_config = payload.get("model_config", {})
    missing = [key for key in REQUIRED_METADATA + STRICT_TRAINING_METADATA if key not in metadata]
    if missing:
        raise ValueError("invalid revised checkpoint; missing metadata: " + ", ".join(missing))
    mismatches = []
    for key, value in (expected or {}).items():
        if key not in metadata:
            mismatches.append("{} missing".format(key))
        elif isinstance(value, float):
            if abs(float(metadata[key]) - value) > 1e-9:
                mismatches.append("{}={} (expected {})".format(key, metadata[key], value))
        elif metadata[key] != value:
            mismatches.append("{}={} (expected {})".format(key, metadata[key], value))
    if mismatches:
        raise ValueError("checkpoint configuration mismatch: " + "; ".join(mismatches))
    if metadata.get("port_selection") or metadata.get("fluid_antennas") or not metadata.get("ideal_csi"):
        raise ValueError("checkpoint is not a fixed-antenna ideal-CSI checkpoint")
    if metadata.get("model_kind") == "iaidnn":
        recorded_architecture = model_config.get(
            "architecture", metadata.get("architecture", "legacy_wmmse_xyzo")
        )
        model_architecture = getattr(model, "architecture", None)
        if model_architecture is not None and str(model_architecture) != str(recorded_architecture):
            raise ValueError(
                "checkpoint architecture mismatch: {} (expected {})".format(
                    recorded_architecture, model_architecture
                )
            )
        if (
            "architecture" in metadata
            and str(recorded_architecture) != str(metadata["architecture"])
        ):
            raise ValueError(
                "checkpoint metadata/model_config architecture mismatch: {} vs {}".format(
                    metadata["architecture"], recorded_architecture
                )
            )
    strict_expected = {
        "training_samples": 50000,
        "epochs": 100,
        "batch_size": 200,
        "train_size": 0.8,
        "learning_rate": 5e-3,
        "reference_update_max_iter": 100,
        "default_inference_layers": 6,
    }
    mismatches = []
    if metadata.get("artifact_scope") not in {"small_sample", "early_stopped"}:
        for key, value in strict_expected.items():
            actual = metadata.get(key)
            if isinstance(value, float):
                valid = abs(float(actual) - value) <= 1e-9
            else:
                valid = actual == value
            if not valid:
                mismatches.append("{}={} (expected {})".format(key, actual, value))
    supervisor = metadata.get("supervisor_epoch_num")
    is_small_sample = metadata.get("artifact_scope") in {"small_sample", "early_stopped"}
    if metadata.get("model_kind") == "deep_mm":
        if abs(float(supervisor) - 0.20) > 1e-9:
            mismatches.append("supervisor_epoch_num={} (expected 0.2)".format(supervisor))
        if not is_small_sample and metadata.get("layer_count") not in SUPPORTED_DEEP_MM_LAYER_COUNTS:
            mismatches.append("layer_count={} (expected 1 through 16)".format(metadata.get("layer_count")))
    elif metadata.get("model_kind") == "gnn":
        if abs(float(supervisor) - 0.0) > 1e-9:
            mismatches.append("supervisor_epoch_num={} (expected 0.0)".format(supervisor))
        if not is_small_sample and metadata.get("layer_count") != 1:
            mismatches.append("layer_count={} (expected 1)".format(metadata.get("layer_count")))
    elif metadata.get("model_kind") == "iaidnn":
        if abs(float(supervisor) - 0.0) > 1e-9:
            mismatches.append("supervisor_epoch_num={} (expected 0.0)".format(supervisor))
        if not is_small_sample and metadata.get("layer_count") != 16:
            mismatches.append("layer_count={} (expected 16)".format(metadata.get("layer_count")))
    else:
        mismatches.append("unknown model_kind={}".format(metadata.get("model_kind")))
    if mismatches:
        raise ValueError("checkpoint strict training configuration mismatch: " + "; ".join(mismatches))
    if (
        metadata.get("artifact_scope") not in {"small_sample", "early_stopped"}
        and metadata.get("validation_enabled") is False
        and int(payload.get("epoch", 0)) != int(metadata.get("epochs", 0))
    ):
        raise ValueError(
            "checkpoint is marked no-validation but was not saved at the final epoch: "
            "{} of {}".format(payload.get("epoch", 0), metadata.get("epochs"))
        )
    try:
        model.load_state_dict(payload["state_dict"])
    except RuntimeError as exc:
        raise ValueError(
            "checkpoint architecture mismatch for {}: {}".format(metadata.get("model_kind"), exc)
        ) from exc
    return payload

