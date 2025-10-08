# registry + factory
from typing import Protocol, List, Dict, Optional, Callable

_REGISTRY: Dict[str, Callable[..., "BaseEmbedder"]] = {}


class BaseEmbedder(Protocol):
    def embed(self, texts: List[str]) -> "np.ndarray": ...  # pragma: no cover


def register(name: str):
    """Decorator để đăng ký embedder vào registry."""
    def _wrap(cls):
        _REGISTRY[name] = cls
        return cls
    return _wrap


def available_backends() -> str:
    return ", ".join(sorted(_REGISTRY.keys()))


def get_embedder(name: str, **kwargs) -> BaseEmbedder:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown dense backend: {name}. Available: {available_backends()}")
    return cls(**kwargs)  # type: ignore


# tiện debug nhanh
def _debug_registry():
    return dict(_REGISTRY)
