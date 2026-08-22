from .frozen_encoder import FrozenFeatureEncoder

__all__ = ["FrozenFeatureEncoder", "TimmViTEncoder"]


def __getattr__(name: str):
    """Lazy import so environments without timm can still use the core modules."""
    if name == "TimmViTEncoder":
        from .vit_timm import TimmViTEncoder

        return TimmViTEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
