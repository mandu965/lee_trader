from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRODUCTION_CONFIG_PATH = ROOT / "config" / "production_v1.yaml"


def resolve_production_config_path() -> Path:
    raw = str(os.environ.get("LEE_PRODUCTION_CONFIG") or "").strip()
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path
    return DEFAULT_PRODUCTION_CONFIG_PATH


@lru_cache(maxsize=1)
def load_production_config() -> dict[str, Any]:
    path = resolve_production_config_path()
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"production config must be a mapping: {path}")
    return payload


def get_production_config_value(path: list[str], default: Any = None) -> Any:
    node: Any = load_production_config()
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def get_production_versions() -> dict[str, str]:
    metadata = get_production_config_value(["metadata"], {}) or {}
    if not isinstance(metadata, dict):
        return {}
    return {str(key): str(value) for key, value in metadata.items() if value is not None}


def get_runtime_mode(default: str = "research") -> str:
    raw = str(
        os.environ.get("LEE_TRADER_RUNTIME_MODE")
        or get_production_config_value(["runtime", "mode"], default)
        or default
    ).strip().lower()
    return raw or default


def is_operational_runtime_mode() -> bool:
    return get_runtime_mode() == "operational"


def allow_experimental_runtime_features(default: bool = True) -> bool:
    env_raw = os.environ.get("LEE_ALLOW_EXPERIMENTAL_FEATURES")
    if env_raw is not None:
        return str(env_raw).strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    configured = get_production_config_value(["runtime", "allow_experimental_features"], default)
    return bool(configured)


_SCORE_FORMULA_VERSION_DEFAULT = "ranking_builder_v8_return_prob_tech_regime"


def get_score_formula_version() -> str:
    """Returns the active score formula version string.

    Priority: SCORE_FORMULA_VERSION env var > config metadata.score_formula_version > built-in default.

    The env var acts as a feature flag for switching formula versions.
    In operational runtime mode, ranking_builder.resolve_score_formula_version() ignores the
    env var override for safety — the switch is prepared here but left off by default.
    To activate in operational mode, set SCORE_FORMULA_VERSION explicitly in .env AND
    update ranking_builder.resolve_score_formula_version() to honour it.
    """
    env_raw = str(os.environ.get("SCORE_FORMULA_VERSION") or "").strip()
    if env_raw:
        return env_raw
    configured = get_production_config_value(["metadata", "score_formula_version"], _SCORE_FORMULA_VERSION_DEFAULT)
    return str(configured or _SCORE_FORMULA_VERSION_DEFAULT)
