"""Abstract base class for satellite/aerial image providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ImageProvider(ABC):
    """Interface for fetching satellite or aerial imagery at a given zoom level."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'google_static', 'naip')."""

    @property
    @abstractmethod
    def min_zoom(self) -> int:
        """Lowest Google Maps zoom level this provider supports."""

    @property
    @abstractmethod
    def max_zoom(self) -> int:
        """Highest Google Maps zoom level this provider supports."""

    def supports_zoom(self, zoom: int) -> bool:
        """Whether this provider serves imagery at the given zoom level."""
        return self.min_zoom <= zoom <= self.max_zoom

    @abstractmethod
    async def fetch(
        self,
        center: tuple[float, float],
        zoom: int,
        output_path: str,
        image_size: int = 640,
    ) -> str:
        """Fetch an image and save to output_path. Return the path on success."""
