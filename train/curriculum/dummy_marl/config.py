from typing import Any, Literal

from pydantic import BaseModel, Field

from schema.policy import DEFAULT_SSP_AGENT_IMAGE
from utils.config import ServerConfig


class DummyMarlCurriculumConfig(BaseModel):
    type: Literal["dummy_marl"] = "dummy_marl"
    debug: bool = False
    team_side: Literal["left", "right", "rand"] = "left"

    grpc_server: ServerConfig = Field(default_factory=lambda: ServerConfig())

    our_goalie_unum: int | None = Field(default=1, ge=1, le=11)
    oppo_goalie_unum: int | None = Field(default=1, ge=1, le=11)

    time_up: int = Field(default=5000, ge=0, le=65535)
    goal_l: int | None = Field(default=None, ge=0, le=255)
    goal_r: int | None = Field(default=None, ge=0, le=255)

    our_team_name: str = "nexus-prime"
    oppo_team_name: str = "bot"

    player_agent_image: str = DEFAULT_SSP_AGENT_IMAGE
    player_bot_image: str = "HELIOS/helios-base"

    def model_post_init(self, context: Any, /) -> None:
        if self.our_goalie_unum is not None and self.our_goalie_unum > 11:
            raise ValueError("our_goalie_unum must be in the 11vs11 roster")
        if self.oppo_goalie_unum is not None and self.oppo_goalie_unum > 11:
            raise ValueError("oppo_goalie_unum must be in the 11vs11 roster")
