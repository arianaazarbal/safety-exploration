"""textnorm: small text normalization utilities (slugs, whitespace, truncation)."""

from .core import collapse_whitespace, slugify, truncate

__version__ = "0.3.1"

__all__ = ["slugify", "collapse_whitespace", "truncate", "__version__"]
