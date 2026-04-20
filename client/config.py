from pydantic import BaseModel

class ClientConfig(BaseModel):
    base_url: str
    timeout_s: float = 10
