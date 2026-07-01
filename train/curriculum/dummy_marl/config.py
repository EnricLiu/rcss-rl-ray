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

    # 11-player SSP sidecars can occasionally skip a player frame while the
    # remaining players are blocked waiting for actions.  Keep reset strict,
    # but allow in-episode partial state batches to reuse recent cached states.
    allow_stale_agent_states: bool = True
    state_fetch_timeout_s: float = Field(default=5.0, gt=0.0)
    reset_state_fetch_timeout_s: float = Field(default=180.0, gt=0.0)
    truth_fetch_timeout_s: float = Field(default=30.0, gt=0.0)
    partial_state_min_unums: int = Field(default=1, ge=1, le=11)
    max_stale_state_cycles: int = Field(default=20, ge=0)

    our_team_name: str = "nexus-prime"
    oppo_team_name: str = "bot"

    player_agent_image: str = DEFAULT_SSP_AGENT_IMAGE
    player_bot_image: str = "HELIOS/helios-base"

    def model_post_init(self, context: Any, /) -> None:
        if self.our_goalie_unum is not None and self.our_goalie_unum > 11:
            raise ValueError("our_goalie_unum must be in the 11vs11 roster")
        if self.oppo_goalie_unum is not None and self.oppo_goalie_unum > 11:
            raise ValueError("oppo_goalie_unum must be in the 11vs11 roster")
