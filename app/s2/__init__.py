"""S2 pipeline package for CAD topology deduplication."""

from .pipeline import S2Pipeline, main
from .settings import S2PipelineConfig

__all__ = [
    "S2Pipeline",
    "S2PipelineConfig",
    "main",
]
