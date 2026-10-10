from pathlib import Path


def project_root() -> Path:
    """src/utils/system.py → src/utils → src → project_root"""
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path: Path | str) -> Path:
    """Resolve a relative path from project_root() (absolute paths are kept)."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = project_root() / resolved
    return resolved


def ensure_output_path(path: Path | str) -> Path:
    """Resolve a relative path from project_root() and create its parent directory."""
    output_path = resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
