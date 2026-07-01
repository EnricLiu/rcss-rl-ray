from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rcss_env import obs as observation
from rcss_env.grpc_srv.proto import pb2
from schema import TeamSide


@dataclass(frozen=True)
class AgentIndexEntry:
    side: TeamSide
    unum: int
    team_name: str
    bot_image: str

    def to_json(self) -> dict[str, object]:
        return {
            "side": self.side.value,
            "unum": self.unum,
            "team_name": self.team_name,
            "bot_image": self.bot_image,
        }


def team_agent_index(*, side: TeamSide, team_name: str, bot_image: str) -> list[AgentIndexEntry]:
    return [
        AgentIndexEntry(
            side=side,
            unum=unum,
            team_name=team_name,
            bot_image=bot_image,
        )
        for unum in range(1, 12)
    ]


def _side_to_pb(side: TeamSide) -> pb2.Side:
    if side == TeamSide.LEFT:
        return pb2.LEFT
    if side == TeamSide.RIGHT:
        return pb2.RIGHT
    raise ValueError(f"Unsupported side: {side!r}")


def _copy_common_fields(src: object, dst: object) -> None:
    dst_fields = {field.name: field for field in dst.DESCRIPTOR.fields}
    for src_field in src.DESCRIPTOR.fields:
        dst_field = dst_fields.get(src_field.name)
        if dst_field is None or src_field.is_repeated:
            continue
        value = getattr(src, src_field.name)
        if src_field.message_type is not None:
            getattr(dst, src_field.name).CopyFrom(value)
        else:
            setattr(dst, src_field.name, value)


def _player_maps_for_side(wm: pb2.WorldModel, side: TeamSide) -> tuple[object, object]:
    side_value = _side_to_pb(side)
    if int(wm.our_side) in (0, side_value):
        return wm.our_players_dict, wm.their_players_dict
    return wm.their_players_dict, wm.our_players_dict


def _set_ball_relative_to_self(wm: pb2.WorldModel) -> None:
    dx = float(wm.ball.position.x - wm.self.position.x)
    dy = float(wm.ball.position.y - wm.self.position.y)
    wm.ball.relative_position.x = dx
    wm.ball.relative_position.y = dy
    wm.ball.dist_from_self = float(math.hypot(dx, dy))
    wm.ball.angle_from_self = float(math.degrees(math.atan2(dy, dx)))
    wm.self.dist_from_ball = wm.ball.dist_from_self
    wm.self.angle_from_ball = wm.ball.angle_from_self


def project_global_world_model(
    wm: pb2.WorldModel,
    *,
    side: TeamSide,
    unum: int,
) -> pb2.WorldModel:
    """Project a trainer global WorldModel to a player-centric WorldModel."""
    our_players, their_players = _player_maps_for_side(wm, side)
    if unum not in our_players:
        raise ValueError(
            f"Cannot project trainer world model cycle={wm.cycle}: "
            f"side={side.value} unum={unum} is not present in own player map"
            f"wm={wm!r}"
        )

    projected = pb2.WorldModel()
    projected.CopyFrom(wm)
    projected.our_side = _side_to_pb(side)

    if projected.our_side != wm.our_side and int(wm.our_side) != 0:
        projected.our_team_name = wm.their_team_name
        projected.their_team_name = wm.our_team_name
        projected.our_goalie_uniform_number = wm.their_goalie_uniform_number
        projected.their_goalie_uniform_number = wm.our_goalie_uniform_number
        projected.our_team_score = wm.their_team_score
        projected.their_team_score = wm.our_team_score
        projected.is_our_set_play = wm.is_their_set_play
        projected.is_their_set_play = wm.is_our_set_play

    projected.ClearField("self")
    _copy_common_fields(our_players[unum], projected.self)
    projected.self.side = _side_to_pb(side)
    projected.self.uniform_number = unum

    projected.ClearField("our_players_dict")
    projected.ClearField("their_players_dict")
    for player_unum, player in our_players.items():
        projected.our_players_dict[int(player_unum)].CopyFrom(player)
    for player_unum, player in their_players.items():
        projected.their_players_dict[int(player_unum)].CopyFrom(player)

    projected.ClearField("teammates")
    projected.ClearField("opponents")
    for player in our_players.values():
        projected.teammates.add().CopyFrom(player)
    for player in their_players.values():
        projected.opponents.add().CopyFrom(player)

    _set_ball_relative_to_self(projected)
    return projected


def missing_projectable_agents(
    wm: pb2.WorldModel,
    agent_index: list[AgentIndexEntry],
) -> list[AgentIndexEntry]:
    missing: list[AgentIndexEntry] = []
    for entry in agent_index:
        our_players, _ = _player_maps_for_side(wm, entry.side)
        if entry.unum not in our_players:
            missing.append(entry)
    return missing


def extract_agent_obs(wm: pb2.WorldModel, entry: AgentIndexEntry) -> np.ndarray:
    projected = project_global_world_model(wm, side=entry.side, unum=entry.unum)
    obs = np.asarray(observation.extract(projected), dtype=np.float32)
    expected_shape = (observation.dim(),)
    if obs.shape != expected_shape:
        raise ValueError(f"Observation shape mismatch: expected={expected_shape}, got={obs.shape}")
    if not np.isfinite(obs).all():
        obs = np.nan_to_num(obs, copy=False)
    return obs
