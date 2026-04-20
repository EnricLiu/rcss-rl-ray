from __future__ import annotations

import argparse
from ipaddress import IPv4Address
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.room.info import RoomInfo
from main import build_actor_options, build_smoke_request, json_safe, summarize_agent_payload


def test_build_actor_options_adds_node_resource_when_target_ip_is_set() -> None:
    args = argparse.Namespace(remote_num_cpus=1.5, target_node_ip="10.42.0.17")

    options = build_actor_options(args)

    assert options == {
        "num_cpus": 1.5,
        "resources": {"node:10.42.0.17": 0.001},
    }



def test_build_smoke_request_keeps_optional_grpc_host_only_when_provided() -> None:
    args = argparse.Namespace(
        num_agents=1,
        grpc_port=50051,
        allocator_host="allocator.default.svc",
        allocator_port=8080,
        bot_image="HELIOS/helios-base",
        agent_image="Cyrus2D/SoccerSimulationProxy",
        time_up=200,
        steps=1,
        grpc_host=None,
    )

    request = build_smoke_request(args)

    assert request == {
        "num_agents": 1,
        "grpc_port": 50051,
        "allocator_host": "allocator.default.svc",
        "allocator_port": 8080,
        "bot_image": "HELIOS/helios-base",
        "agent_image": "Cyrus2D/SoccerSimulationProxy",
        "time_up": 200,
        "steps": 1,
    }



def test_json_safe_handles_pydantic_models_and_numpy_values() -> None:
    room = RoomInfo(name="room-a", host=IPv4Address("127.0.0.1"), ports={"default": 6666, "mc": 7777})

    payload = json_safe(
        {
            "room": room,
            "rewards": np.array([1.0, 2.0], dtype=np.float32),
            "scalar": np.float32(3.5),
        }
    )

    assert payload == {
        "room": {
            "name": "room-a",
            "host": "127.0.0.1",
            "ports": {"default": 6666, "mc": 7777},
        },
        "rewards": [1.0, 2.0],
        "scalar": 3.5,
    }



def test_summarize_agent_payload_describes_masked_observation_mapping() -> None:
    summary = summarize_agent_payload(
        {
            "obs": np.zeros((124,), dtype=np.float32),
            "act_mask": np.array([1, 0, 1, 1], dtype=np.int8),
        }
    )

    assert summary == {
        "kind": "mapping",
        "keys": ["obs", "act_mask"],
        "obs_shape": [124],
        "obs_dtype": "float32",
        "act_mask_shape": [4],
        "act_mask_active": 3,
    }




