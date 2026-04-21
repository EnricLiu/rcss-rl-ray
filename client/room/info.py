from ipaddress import IPv4Address
from pydantic import BaseModel

POD_PORT_MAPPING = {
    "default": 6666,
    "player": 6657,
    "trainer": 6658,
    "coach": 6659,
    "player-raw": 6000,
    "trainer-raw": 6001,
    "coach-raw": 6002,
    "mc": 7777,
}

class RoomInfo(BaseModel):
    name: str
    pod: IPv4Address
    host: IPv4Address
    ports: dict[str, int]

    def base_url(self, key: str) -> str:
        if key not in POD_PORT_MAPPING:
            raise KeyError(f"Room port '{key}' is not supported in POD_PORT_MAPPING")
        return f"http://{self.pod}:{POD_PORT_MAPPING[key]}"

    def host_base_url(self, key: str) -> str:
        if key not in self.ports:
            raise KeyError(f"Room port '{key}' is not available in allocation result")
        return f"http://{self.host}:{self.ports[key]}"

    @property
    def base_url_rcss(self) -> str:
        return self.base_url("default")

    @property
    def base_url_mc(self) -> str:
        return self.base_url("mc")

    @property
    def host_base_url_rcss(self) -> str:
        return self.host_base_url("default")

    @property
    def host_base_url_mc(self) -> str:
        return self.host_base_url("mc")
