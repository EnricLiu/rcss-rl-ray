from typing import Literal, Any, Optional

from pydantic import BaseModel, Field
from utils.config import ServerConfig

from ..utils import Region


class ShootingCurriculumConfig(BaseModel):
    debug: bool = False
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

    # ---- Sparse / event-based ----
    reward_goal: float = Field(default=10.0, ge=0.0)
    reward_concede: float = Field(default=10.0, ge=0.0)
    reward_out_of_bounds: float = Field(default=1.0, ge=0.0)
    # One-shot bonus on the rising edge of `self.is_kickable` (agent first
    # touches / gains control of the ball). Critical to bootstrap "approach the
    # ball" behaviour together with the agent->ball shaping below.
    reward_kickable_bonus: float = Field(default=0.5, ge=0.0)

    # ---- Potential-based shaping (PBRS, gamma-aware) ----
    # Encourages the AGENT to approach the BALL.  This is the key term that
    # was missing previously and is required for the agent to even learn to
    # close in on the ball.
    reward_agent_to_ball_shaping: float = Field(default=1.0, ge=0.0)
    # Encourages the BALL to approach the opponent goal.
    reward_ball_to_goal_shaping: float = Field(default=1.0, ge=0.0)
    # Discount factor used in gamma-aware PBRS (`gamma * Phi' - Phi`).
    # Should match (or be close to) the RL training discount.
    gamma_shaping: float = Field(default=0.99, ge=0.0, le=1.0)
    # Per-step clip applied to each shaping component to suppress spikes
    # caused by referee resets / position teleports.
    shaping_clip: float = Field(default=0.1, gt=0.0)

    # ---- Dense action-quality signals ----
    # Reward (per cycle) proportional to ball-velocity component along the
    # direction from ball to opponent goal centre, normalized to [-1, 1].
    reward_ball_velocity_to_goal: float = Field(default=0.05, ge=0.0)

    # ---- Step costs ----
    reward_time_decay: float = Field(default=0.001, ge=0.0)
    # Hard cap on Δcycle used by time decay & shaping; prevents huge penalties
    # / spikes when a stoppage skips many cycles between two PlayOn frames.
    max_cycle_gap: int = Field(default=5, ge=1)

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
