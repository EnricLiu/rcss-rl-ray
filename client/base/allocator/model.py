from typing import Literal

from pydantic import BaseModel
from schema import GameServerSchema

class PostAllocateRoomRequest(BaseModel):
    conf: GameServerSchema
    version: int = 1

class PostCreateFleetRequest(BaseModel):
    name: str
    conf: GameServerSchema
    version: int = 1

class DeleteDropFleetRequest(BaseModel):
    name: str

class GetFleetTemplateRequest(BaseModel):
    format: Literal["json", "yaml"] = "json"
    