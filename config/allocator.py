from pydantic import BaseModel


class AllocatorConfig(BaseModel):
    base_url: str
    timeout_s: int = 10
