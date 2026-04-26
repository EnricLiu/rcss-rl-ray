from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from httpx import Client, MockTransport, Request, Response

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.base.allocator import AllocatorClient, AllocatorConfig
from client.base.http import ClusterApiError, unwrap_response
from client.base.mc import MatchComposerClient
from client.base.rcss import RcssServerClient
from client.room import RoomClient
from schema import GameServerSchema, PlayerSchema, TeamSchema, TeamSide, TeamsSchema


def make_schema() -> GameServerSchema:
    return GameServerSchema(
        teams=TeamsSchema(
            left=TeamSchema(
                name="left-team",
                side=TeamSide.LEFT,
                players=[PlayerSchema(unum=1)],
            ),
            right=TeamSchema(
                name="right-team",
                side=TeamSide.RIGHT,
                players=[PlayerSchema(unum=1)],
            ),
        )
    )


def envelope(payload: Any, *, success: bool = True) -> dict[str, Any]:
    return {
        "id": 1,
        "success": success,
        "payload": payload,
        "created_at": "2026-04-19T00:00:00Z",
    }


def test_unwrap_response_raises_on_success_false() -> None:
    response = Response(
        200,
        json=envelope({"error": "Shutdown Failed", "desc": "service running"}, success=False),
    )

    try:
        unwrap_response(response)
    except ClusterApiError as exc:
        assert "Shutdown Failed" in str(exc)
    else:
        raise AssertionError("Expected ClusterApiError")


def test_allocator_request_room_matches_server_contract() -> None:
    def handler(request: Request) -> Response:
        assert request.method == "POST"
        assert request.url.path == "/gs/allocate"

        body = json.loads(request.content.decode())
        assert body["version"] == 1
        assert "conf" in body
        assert "teams" in body["conf"]

        return Response(
            200,
            json=envelope(
                {
                    "name": "room-a",
                    "host": "127.0.0.1",
                    "ports": {"default": 6666, "mc": 7777},
                }
            ),
        )

    client = Client(
        base_url="http://allocator.test",
        transport=MockTransport(handler),
    )
    allocator = AllocatorClient(
        AllocatorConfig(base_url="http://allocator.test"),
        client=client,
    )

    room = allocator.request_room(make_schema())

    assert room.info.name == "room-a"
    assert room.info.base_url_rcss == "http://127.0.0.1:6666"
    assert room.info.base_url_mc == "http://127.0.0.1:7777"


def test_allocator_drop_fleet_uses_delete_json_body() -> None:
    def handler(request: Request) -> Response:
        assert request.method == "DELETE"
        assert request.url.path == "/fleet"
        assert request.url.query == b""
        assert json.loads(request.content.decode()) == {"name": "fleet-a"}
        return Response(200, json=envelope({}))

    client = Client(
        base_url="http://allocator.test",
        transport=MockTransport(handler),
    )
    allocator = AllocatorClient(
        AllocatorConfig(base_url="http://allocator.test"),
        client=client,
    )

    allocator.drop_fleet("fleet-a")


def test_room_release_calls_rcss_shutdown_with_force_true() -> None:
    def handler(request: Request) -> Response:
        assert request.method == "POST"
        assert request.url.path == "/control/shutdown"
        assert json.loads(request.content.decode()) == {"force": True}
        return Response(200, json=envelope(None))

    allocator = AllocatorClient(AllocatorConfig(base_url="http://allocator.test"))
    room = RoomClient(
        {
            "name": "room-a",
            "host": "127.0.0.1",
            "ports": {"default": 6666, "mc": 7777},
        },
        allocator,
    )
    room._RoomClient__rcss = RcssServerClient(
        "http://127.0.0.1:6666",
        client=Client(
            base_url="http://127.0.0.1:6666",
            transport=MockTransport(handler),
        ),
    )

    room.release()


def test_match_composer_status_parses_payload() -> None:
    def handler(request: Request) -> Response:
        assert request.method == "GET"
        assert request.url.path == "/status"
        return Response(
            200,
            json=envelope(
                {
                    "in_match": True,
                    "rcss": {"host": "127.0.0.1", "port": 6666},
                    "status": "Idle",
                    "team_l": {
                        "name": "left-team",
                        "side": "left",
                        "status": {"status": "running"},
                        "players": {},
                    },
                    "team_r": {
                        "name": "right-team",
                        "side": "right",
                        "status": {"status": "running"},
                        "players": {},
                    },
                }
            ),
        )

    mc = MatchComposerClient(
        "http://mc.test",
        client=Client(base_url="http://mc.test", transport=MockTransport(handler)),
    )

    status = mc.status()

    assert status.in_match is True
    assert status.info is not None
    assert status.info.team_l.name == "left-team"


def test_rcss_trainer_team_names_uses_tuple_struct_request_shape() -> None:
    def handler(request: Request) -> Response:
        assert request.method == "POST"
        assert request.url.path == "/command/trainer/team_names"
        assert json.loads(request.content.decode()) == [None]
        return Response(
            200,
            json=envelope({"ok": True, "left": "Alpha", "right": "Beta"}),
        )

    rcss = RcssServerClient(
        "http://rcss.test",
        client=Client(
            base_url="http://rcss.test",
            transport=MockTransport(handler),
        ),
    )

    result = rcss.trainer.team_names()

    assert result.ok is True
    assert result.left == "Alpha"
    assert result.right == "Beta"