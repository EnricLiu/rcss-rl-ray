from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import ConfigDict

from schema._base import SchemaModel


class OpenResultModel(SchemaModel):
	model_config = ConfigDict(extra="allow", populate_by_name=True)


class ControlShutdownRequest(SchemaModel):
	force: bool = True


class MetricsStatusResponse(OpenResultModel):
	service: Any
	conn_count: int
	agones: Any | None = None


class MetricsConnectionInfo(OpenResultModel):
	name: str
	status: Any
	touched_at: datetime


class MetricsConfigResponse(OpenResultModel):
	log_root: str
	half_time_auto_start: bool | None = None
	always_log_stdout: bool | None = None
	rcss_game_log_rel_dir: str | None = None
	rcss_stdio_log_rel_path: str | None = None


class TrainerChangeModeRequest(SchemaModel):
	play_mode: str


class TrainerEarRequest(SchemaModel):
	mode: Literal["on", "off"]


class TrainerInitRequest(SchemaModel):
	version: int | None = None


class TrainerCommandResult(OpenResultModel):
	ok: bool


class TrainerCheckBallResponse(TrainerCommandResult):
	time: int | None = None
	position: str | None = None


class TrainerTeamNamesResponse(TrainerCommandResult):
	left: str | None = None
	right: str | None = None
