"""Training and environment configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlayerInitState:
    """Initial state overrides for a single player.

    Matches the ``init_state`` block in the per-player section of
    *template.json*.

    Attributes
    ----------
    pos_x : float | None
        Normalised X position (0–1 range, ``None`` = default).
    pos_y : float | None
        Normalised Y position (0–1 range, ``None`` = default).
    stamina : float | None
        Starting stamina (``None`` = default).
    """

    pos_x: float | None = None
    pos_y: float | None = None
    stamina: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise; omit *None* values."""
        d: dict[str, Any] = {}
        if self.pos_x is not None and self.pos_y is not None:
            d["pos"] = {"x": self.pos_x, "y": self.pos_y}
        if self.stamina is not None:
            d["stamina"] = self.stamina
        return d


@dataclass
class PlayerConfig:
    """Configuration for a single player in a match.

    Mirrors the per-player block in the *template.json* accepted by the
    rcss_cluster match-composer sidecar.

    Attributes
    ----------
    unum : int
        Uniform number (1-11).
    goalie : bool
        Whether this player is a goalie.
    policy_kind : str
        ``"bot"`` for a scripted agent or ``"agent"`` for an RL training
        agent (SoccerSimulationProxy).
    policy_image : str | None
        Docker image name.  For ``"bot"`` this is the bot image
        (e.g. ``"HELIOS/helios-base"``); for ``"agent"`` this is the
        SoccerSimulationProxy image
        (e.g. ``"Cyrus2D/SoccerSimulationProxy"``).
    policy_agent : str | None
        Agent type identifier, used when *policy_kind* is ``"agent"``
        (e.g. ``"ssp"``).
    grpc_host : str | None
        gRPC host address that the sidecar will connect to.
        Only used when *policy_kind* is ``"agent"``.
    grpc_port : int | None
        gRPC port that the sidecar will connect to.
        Only used when *policy_kind* is ``"agent"``.
    init_state : PlayerInitState | None
        Optional initial state overrides (position, stamina).
    blocklist : dict[str, bool] | None
        Optional map of action names to blocked status
        (e.g. ``{"dash": True, "catch": False}``).
    """

    unum: int = 1
    goalie: bool = False
    policy_kind: str = "agent"
    policy_image: str | None = None
    policy_agent: str | None = None
    grpc_host: str | None = None
    grpc_port: int | None = None
    init_state: PlayerInitState | None = None
    blocklist: dict[str, bool] | None = None


def _default_ally_players() -> list[PlayerConfig]:
    """Default allied team: 3 agent players."""
    return [
        PlayerConfig(unum=i, goalie=(i == 1), policy_kind="agent")
        for i in range(1, 4)
    ]


def _default_opponent_players() -> list[PlayerConfig]:
    """Default opponent team: 3 bot players."""
    return [
        PlayerConfig(unum=i, goalie=(i == 1), policy_kind="bot")
        for i in range(1, 4)
    ]


@dataclass
class EnvConfig:
    """Parameters that control the RCSS environment.

    The team composition fields (``ally_players`` / ``opponent_players``)
    follow the `template.json`_ structure used by the rcss_cluster
    allocator.  Each player is individually configurable as a *bot* or an
    RL *agent* via :class:`PlayerConfig`.

    .. _template.json:
       https://github.com/EnricLiu/rcss_cluster/blob/sidecar/
       match_composer/sidecars/match_composer/docs/template.json

    Attributes
    ----------
    ally_team_name : str
        Name of the allied (training) team.
    opponent_team_name : str
        Name of the opponent team.
    ally_players : list[PlayerConfig]
        Player specifications for the allied team (up to 11).
    opponent_players : list[PlayerConfig]
        Player specifications for the opponent team (up to 11).
    time_up : int
        Simulation stop time in server cycles (``stopping.time_up``).
    goal_limit_l : int
        Stop after the left team scores this many goals (0 = disabled).
    referee_enable : bool
        Whether to enable the referee module.
    ball_init_x : float | None
        Normalised initial X position of the ball.
    ball_init_y : float | None
        Normalised initial Y position of the ball.
    max_episode_steps : int
        Training-side episode truncation (may differ from *time_up*).
    mode : str
        ``"local"`` for built-in simulation, ``"remote"`` for
        rcss_cluster allocation.
    allocator_url : str
        Allocator REST endpoint (remote mode only).
    grpc_host : str
        gRPC listen address (remote mode only).
    grpc_port : int
        gRPC listen port (remote mode only).

    The following fields are used **only** by the local simplified
    simulation and are ignored in remote mode:

    field_half_x, move_speed, kick_radius, kick_power,
    goal_reward, distance_penalty, seed.
    """

    # --- Team composition (mirrors template.json) ---
    ally_team_name: str = "RLAgent"
    opponent_team_name: str = "Bot"
    ally_players: list[PlayerConfig] = field(
        default_factory=_default_ally_players,
    )
    opponent_players: list[PlayerConfig] = field(
        default_factory=_default_opponent_players,
    )

    # --- Stopping / referee / init_state (template.json top-level) ---
    time_up: int = 6000
    goal_limit_l: int = 0
    referee_enable: bool = False
    ball_init_x: float | None = None
    ball_init_y: float | None = None

    # --- Training-side truncation ---
    max_episode_steps: int = 200

    # --- Remote mode ---
    mode: str = "local"
    allocator_url: str = ""
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50051

    # --- Local simulation parameters ---
    field_half_x: float = 10.0
    move_speed: float = 0.5
    kick_radius: float = 1.0
    kick_power: float = 2.0
    goal_reward: float = 10.0
    distance_penalty: float = 0.01
    seed: int | None = None

    # --- Derived helpers (not fields) ---

    @property
    def num_left(self) -> int:
        """Number of players on the allied (left) team."""
        return len(self.ally_players)

    @property
    def num_right(self) -> int:
        """Number of players on the opponent (right) team."""
        return len(self.opponent_players)


@dataclass
class TrainConfig:
    """Top-level training configuration passed to RLlib.

    Attributes:
        algo:                RLlib algorithm name (e.g. ``"PPO"``, ``"IMPALA"``).
        num_env_runners:     Number of parallel environment runner workers.
        num_envs_per_runner: Number of vectorised envs per runner worker.
        train_batch_size:    Total number of transitions per SGD update.
        sgd_minibatch_size:  Mini-batch size for each SGD step (PPO only).
        num_sgd_iter:        Number of SGD epochs per training iteration.
        lr:                  Optimiser learning rate.
        gamma:               Discount factor.
        entropy_coeff:       Entropy bonus coefficient.
        clip_param:          PPO clipping parameter.
        num_iterations:      Total number of training iterations to run.
        checkpoint_freq:     Save a checkpoint every N iterations (0 = never).
        checkpoint_dir:      Directory where checkpoints are written.
        env_config:          Environment configuration forwarded to :class:`RCSSEnv`.
    """

    algo: str = "PPO"
    num_env_runners: int = 2
    num_envs_per_runner: int = 1
    train_batch_size: int = 4000
    sgd_minibatch_size: int = 128
    num_sgd_iter: int = 10
    lr: float = 3e-4
    gamma: float = 0.99
    entropy_coeff: float = 0.01
    clip_param: float = 0.3
    num_iterations: int = 100
    checkpoint_freq: int = 10
    checkpoint_dir: str = "checkpoints"
    env_config: EnvConfig = field(default_factory=EnvConfig)
