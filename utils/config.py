from ipaddress import IPv4Address

from pydantic import BaseModel, IPvAnyAddress, Field


class ServerConfig(BaseModel):
    """
    Network service connection configuration.
    Generic server connection settings (host + port + timeout).
    """

    host: IPvAnyAddress = IPv4Address("127.0.0.1")
    port: int = 12345
    timeout: int = 10  # timeout in seconds

    @property
    def addr(self) -> str:
        """Return the address as a ``host:port`` string."""
        return f"{self.host}:{self.port}"
