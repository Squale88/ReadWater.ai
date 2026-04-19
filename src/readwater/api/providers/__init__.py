"""Image provider abstraction layer."""

from readwater.api.providers.base import ImageProvider
from readwater.api.providers.google_static import GoogleStaticProvider
from readwater.api.providers.naip import NAIPProvider
from readwater.api.providers.placeholder import PlaceholderProvider
from readwater.api.providers.registry import ImageProviderRegistry

__all__ = [
    "ImageProvider",
    "GoogleStaticProvider",
    "ImageProviderRegistry",
    "NAIPProvider",
    "PlaceholderProvider",
]
