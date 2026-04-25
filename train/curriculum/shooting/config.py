from typing import Literal, Any, Optional

from pydantic import BaseModel, Field
from utils.config import ServerConfig

from ..utils import Region


class ShootingCurriculumConfig(BaseModel):
    debug: bool = True
    agent_unum: int = Field(ge=1, le=11)
    team_side: Literal["left", "right", "rand"] = "left"

    grpc_server: ServerConfig = Field(default_factory=lambda: ServerConfig())

    our_player_num: int = Field(ge=1, le=11)
    oppo_player_num: int = Field(ge=1, le=11)
    our_goalie_unum: Optional[int] = Field(default=1, ge=1, le=11)
    oppo_goalie_unum: Optional[int] = Field(default=1, ge=1, le=11)

    time_up: int = Field(default=5000, ge=0, le=65535)
    goal_l: Optional[int] = Field(default=1, ge=0, le=255)
    goal_r: Optional[int] = Field(default=1, ge=0, le=255)

    our_team_name: str = "nexus-prime"
    oppo_team_name: str = "bot"

    agent_region: Region = Field(default_factory=lambda: Region.from_range(x=(-20, 30), y=(-30, 30)))
    ball_region: Region = Field(default_factory=lambda: Region.from_range(x=(-20, 40), y=(-34, 34)))

    player_agent_image: str = "Cyrus2D/SoccerSimulationProxy"
    player_bot_image: str = "HELIOS/helios-base"

    def model_post_init(self, context: Any, /) -> None:
        if self.agent_unum > self.our_player_num:
            raise ValueError(
                f"agent_unum={self.agent_unum} must be less equal than our_player_num={self.our_player_num}"
            )
        if self.our_goalie_unum is not None and self.our_goalie_unum > self.our_player_num:
            raise ValueError(
                f"our_goalie_unum={self.our_goalie_unum}, but our_player_num={self.our_player_num}, no unum for goalie"
            )
        if self.oppo_goalie_unum is not None and self.oppo_goalie_unum > self.oppo_player_num:
            raise ValueError(
                f"oppo_goalie_unum={self.oppo_goalie_unum}, but oppo_player_num={self.oppo_player_num}, no unum for goalie"
            )
