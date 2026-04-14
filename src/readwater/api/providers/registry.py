"""Registry mapping roles to image providers."""

from __future__ import annotations

from readwater.api.providers.base import ImageProvider


class ImageProviderRegistry:
    """Maps roles ('overview', 'structure') to image providers."""

    def __init__(self) -> None:
        self._providers: dict[str, list[ImageProvider]] = {}

    def register(self, provider: ImageProvider, roles: list[str]) -> None:
        """Register a provider for one or more roles."""
        for role in roles:
            self._providers.setdefault(role, []).append(provider)

    def get_providers(self, role: str) -> list[ImageProvider]:
        """Return all providers registered for a role. Raises if none."""
        providers = self._providers.get(role, [])
        if not providers:
            raise ValueError(f"No providers registered for role '{role}'")
        return list(providers)

    def get_default_provider(self, role: str) -> ImageProvider:
        """Return the first provider registered for a role."""
        return self.get_providers(role)[0]
