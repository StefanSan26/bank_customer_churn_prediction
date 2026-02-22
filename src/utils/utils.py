"""Small helpers for paths and logging."""
import os


def project_root() -> str:
    """Return absolute path to project root (parent of src)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
