from dataclasses import dataclass

@dataclass
class ServerConfig:
    host: str
    port: int
    timeout: int = 10

    @property
    def addr(self) -> str:
        return f"{self.host}:{self.port}"
