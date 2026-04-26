from pydantic import BaseModel


class FleetInfo(BaseModel):
    name: str
