"""General-purpose eyeglass glare removal package for video."""

from eyeglass_glare_removal.config import (
    CompositeConfig,
    InpaintConfig,
    PipelineConfig,
    PrepConfig,
    ReferenceConfig,
    SegmentConfig,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CompositeConfig",
    "InpaintConfig",
    "PipelineConfig",
    "PrepConfig",
    "ReferenceConfig",
    "SegmentConfig",
]
