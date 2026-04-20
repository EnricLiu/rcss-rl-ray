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
