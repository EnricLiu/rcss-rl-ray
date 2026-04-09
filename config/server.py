"""Network service connection configuration."""

from pydantic import BaseModel


class ServerConfig(BaseModel):
    """Generic server connection settings (host + port + timeout)."""

    host: str
    port: int
    timeout: int = 10  # timeout in seconds

    @property
    def addr(self) -> str:
        """Return the address as a ``host:port`` string."""
        return f"{self.host}:{self.port}"
