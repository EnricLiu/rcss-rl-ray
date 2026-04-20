from ipaddress import IPv4Address
from pydantic import BaseModel


class RoomInfo(BaseModel):
    name: str
    host: IPv4Address
    ports: dict[str, int]

    def base_url(self, key: str) -> str:
        if key not in self.ports:
            raise KeyError(f"Room port '{key}' is not available in allocation result")
        return f"http://{self.host}:{self.ports[key]}"

    @property
    def base_url_rcss(self) -> str:
        return self.base_url("default")

    @property
    def base_url_mc(self) -> str:
        return self.base_url("mc")
