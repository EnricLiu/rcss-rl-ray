from __future__ import annotations

from typing import Any

from pydantic import ConfigDict
from pydantic import model_validator

from schema._base import SchemaModel


class HostPort(SchemaModel):
	host: str
	port: int


class MatchTeamStatus(SchemaModel):
	status: str
	reason: str | None = None


class MatchPlayerInfo(SchemaModel):
	model_config = ConfigDict(extra="allow", populate_by_name=True)

	unum: int
	kind: Any
	status: Any
	image: Any


class MatchTeamInfo(SchemaModel):
	name: str
	side: str
	status: MatchTeamStatus
	players: dict[int, MatchPlayerInfo]


class MatchGameInfo(SchemaModel):
	rcss: HostPort
	status: Any
	team_l: MatchTeamInfo
	team_r: MatchTeamInfo


class MatchStatusResponse(SchemaModel):
	in_match: bool
	info: MatchGameInfo | None = None

	@model_validator(mode="before")
	@classmethod
	def _inflate_flattened_info(cls, value: Any) -> Any:
		if not isinstance(value, dict) or "info" in value:
			return value

		info_keys = {"rcss", "status", "team_l", "team_r"}
		info = {key: value[key] for key in info_keys if key in value}
		if not info:
			return value

		payload = dict(value)
		for key in info:
			payload.pop(key, None)
		payload["info"] = info
		return payload
