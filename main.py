from __future__ import annotations

import argparse
import json
import logging
import traceback
from collections.abc import Mapping, Sequence
from ipaddress import IPv4Address
from typing import Any, cast

import numpy as np
import ray
from pydantic import BaseModel
from ray.util import get_node_ip_address

from rcss_env.env import RCSSEnv
from train import make_default_room_schema, make_env_config

logger = logging.getLogger(__name__)


def json_safe(value: Any) -> Any:
	"""Convert nested objects into JSON-serialisable values."""
	if isinstance(value, BaseModel):
		return value.model_dump(mode="json", by_alias=True)
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, Mapping):
		return {str(k): json_safe(v) for k, v in value.items()}
	if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
		return [json_safe(v) for v in value]
	return value


def summarize_agent_payload(payload: Any) -> dict[str, Any]:
	"""Describe an observation-like payload without dumping full tensors."""
	if isinstance(payload, np.ndarray):
		return {
			"kind": "ndarray",
			"shape": list(payload.shape),
			"dtype": str(payload.dtype),
		}

	if isinstance(payload, Mapping):
		summary: dict[str, Any] = {"kind": "mapping", "keys": list(payload.keys())}
		if "obs" in payload and isinstance(payload["obs"], np.ndarray):
			summary["obs_shape"] = list(payload["obs"].shape)
			summary["obs_dtype"] = str(payload["obs"].dtype)
		if "act_mask" in payload and isinstance(payload["act_mask"], np.ndarray):
			summary["act_mask_shape"] = list(payload["act_mask"].shape)
			summary["act_mask_active"] = int(np.asarray(payload["act_mask"]).sum())
		return summary

	return {"kind": type(payload).__name__, "repr": repr(payload)}


def summarize_observations(obs: Mapping[int, Any]) -> dict[str, Any]:
	return {str(agent_id): summarize_agent_payload(agent_obs) for agent_id, agent_obs in obs.items()}


def build_action_dict(env: RCSSEnv, active_agents: Sequence[int] | None = None) -> dict[int, Any]:
	agent_ids = list(active_agents) if active_agents is not None else list(env.agents)
	return {agent_id: env.action_spaces[agent_id].sample() for agent_id in agent_ids}


def collect_room_diagnostics(env: RCSSEnv) -> dict[str, Any]:
	diagnostics: dict[str, Any] = {
		"room": json_safe(env.room.info),
		"base_url_rcss": env.room.info.base_url_rcss,
		"base_url_mc": env.room.info.base_url_mc,
	}

	probes: dict[str, Any] = {
		"mc_status": env.room.mc.status,
		"rcss_metrics_status": env.room.rcss.metrics_status,
		"rcss_metrics_health": env.room.rcss.metrics_health,
		"rcss_metrics_conn": env.room.rcss.metrics_conn,
		"trainer_team_names": env.room.rcss.trainer.team_names,
	}

	for name, probe in probes.items():
		try:
			diagnostics[name] = json_safe(probe())
		except Exception as exc:  # pragma: no cover - exercised against live cluster
			diagnostics[name] = {
				"error_type": type(exc).__name__,
				"message": str(exc),
			}

	return diagnostics


def build_smoke_request(args: argparse.Namespace) -> dict[str, Any]:
	request = {
		"num_agents": args.num_agents,
		"grpc_port": args.grpc_port,
		"allocator_host": args.allocator_host,
		"allocator_port": args.allocator_port,
		"bot_image": args.bot_image,
		"agent_image": args.agent_image,
		"time_up": args.time_up,
		"steps": args.steps,
	}

	if args.grpc_host:
		request["grpc_host"] = args.grpc_host

	return request


def run_env_smoke(request: Mapping[str, Any]) -> dict[str, Any]:
	grpc_host = str(request.get("grpc_host") or get_node_ip_address())
	grpc_ip = IPv4Address(grpc_host)
	num_agents = int(request["num_agents"])
	grpc_port = int(request["grpc_port"])
	allocator_host = str(request["allocator_host"])
	allocator_port = int(request["allocator_port"])
	bot_image = str(request["bot_image"])
	agent_image = str(request["agent_image"])
	time_up = int(request["time_up"])
	steps = int(request["steps"])

	room_schema = make_default_room_schema(
		num_agents=num_agents,
		grpc_host=grpc_ip,
		grpc_port=grpc_port,
		bot_image=bot_image,
		agent_image=agent_image,
		time_up=time_up,
	)
	env_config = make_env_config(
		grpc_host=grpc_ip,
		grpc_port=grpc_port,
		allocator_host=allocator_host,
		allocator_port=allocator_port,
		gs_schema=room_schema,
	)

	env = RCSSEnv(env_config)
	result: dict[str, Any] = {
		"success": False,
		"grpc_host": grpc_host,
		"grpc_port": grpc_port,
		"allocator_base_url": env.config.allocator.base_url,
		"agent_unums": list(env.agents),
		"steps_requested": steps,
		"schema_summary": {
			"agent_team": env.schema.teams.agent_team.name,
			"left_team": env.schema.teams.left.name,
			"right_team": env.schema.teams.right.name,
			"time_up": env.schema.stopping.time_up,
		},
	}

	try:
		print("version:", env.allocator.fleet_get_template_version())
		result["allocator_health"] = env.allocator.health_check()
		result["allocator_ready"] = env.allocator.readiness_check()

		obs, infos = env.reset()
		result["reset"] = {
			"obs_agents": sorted(obs.keys()),
			"obs_summary": summarize_observations(obs),
			"infos": json_safe(infos),
		}
		result["post_reset_diagnostics"] = collect_room_diagnostics(env)

		step_records: list[dict[str, Any]] = []
		active_agents: list[int] = [int(agent_id) for agent_id in sorted(obs.keys())] if obs else [int(agent_id) for agent_id in env.agents]
		current_obs = obs

		for step_idx in range(steps):
			actions = build_action_dict(env, active_agents)
			next_obs, rewards, terminateds, truncateds, next_infos = env.step(actions)
			step_records.append(
				{
					"step_index": step_idx + 1,
					"action_agents": sorted(actions.keys()),
					"reward_sum": float(sum(rewards.values())),
					"rewards": json_safe(rewards),
					"terminateds": json_safe(terminateds),
					"truncateds": json_safe(truncateds),
					"infos": json_safe(next_infos),
					"obs_summary": summarize_observations(next_obs),
				}
			)
			current_obs = next_obs
			active_agents = [int(agent_id) for agent_id in sorted(current_obs.keys())] if current_obs else active_agents

			if terminateds.get("__all__") or truncateds.get("__all__"):
				break

		result["steps"] = step_records
		result["success"] = True
		return result
	except Exception as exc:  # pragma: no cover - exercised against live cluster
		result["error"] = {
			"type": type(exc).__name__,
			"message": str(exc),
			"traceback": traceback.format_exc(),
		}
		if env.has_room():
			try:
				result["failure_room"] = json_safe(env.room.info)
			except Exception:
				pass
		return result
	finally:
		try:
			env.close()
		except Exception as exc:  # pragma: no cover - exercised against live cluster
			result["close_error"] = {
				"type": type(exc).__name__,
				"message": str(exc),
			}


@ray.remote
class EnvSmokeActor:
	def run(self, request: Mapping[str, Any]) -> dict[str, Any]:
		detected_ip = get_node_ip_address()
		payload = dict(request)
		payload.setdefault("grpc_host", detected_ip)

		result = run_env_smoke(payload)
		result["ray_runtime"] = {
			"mode": "ray_actor",
			"detected_node_ip": detected_ip,
		}
		return result


def describe_ray_cluster() -> dict[str, Any]:
	nodes = []
	for node in ray.nodes():
		if not node.get("Alive"):
			continue
		resources = node.get("Resources", {})
		nodes.append(
			{
				"node_ip": node.get("NodeManagerAddress"),
				"resources": sorted(resources.keys()),
				"is_head": any(key.startswith("node:__internal_head__") for key in resources),
			}
		)

	return {
		"cluster_resources": json_safe(ray.cluster_resources()),
		"available_resources": json_safe(ray.available_resources()),
		"alive_nodes": nodes,
	}


def build_actor_options(args: argparse.Namespace) -> dict[str, Any]:
	options: dict[str, Any] = {
		"num_cpus": args.remote_num_cpus,
	}

	if args.target_node_ip:
		options["resources"] = {f"node:{args.target_node_ip}": 0.001}

	return options


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Smoke-test one RCSSEnv against allocator + RCSS sidecars before training"
	)

	parser.add_argument("--execution-mode", choices=["ray", "local"], default="ray")
	parser.add_argument(
		"--ray-address",
		type=str,
		default="auto",
		help="Ray address, e.g. auto or ray://<head-svc>:10001",
	)
	parser.add_argument(
		"--target-node-ip",
		type=str,
		default=None,
		help="Schedule the smoke actor onto a specific Ray node via resource key node:<ip>",
	)
	parser.add_argument(
		"--remote-num-cpus",
		type=float,
		default=1.0,
		help="CPU reservation for the remote smoke actor",
	)

	parser.add_argument(
		"--grpc-host",
		type=str,
		default=None,
		help="Advertised gRPC host for sidecars. In ray mode defaults to the selected worker pod IP.",
	)
	parser.add_argument("--grpc-port", type=int, default=50051)
	parser.add_argument("--allocator-host", type=str, default="rcss-env-allocator.rcss-gateway-dev.svc.cluster.local")
	parser.add_argument("--allocator-port", type=int, default=80)

	parser.add_argument("--num-agents", type=int, default=1)
	parser.add_argument("--steps", type=int, default=1)
	parser.add_argument("--time-up", type=int, default=200)
	parser.add_argument("--bot-image", type=str, default="HELIOS/helios-base")
	parser.add_argument("--agent-image", type=str, default="Cyrus2D/SoccerSimulationProxy")
	parser.add_argument(
		"--log-level",
		type=str,
		default="INFO",
		choices=["DEBUG", "INFO", "WARNING", "ERROR"],
	)

	return parser.parse_args(argv)


def run_remote_smoke(args: argparse.Namespace, request: Mapping[str, Any]) -> dict[str, Any]:
	init_kwargs: dict[str, Any] = {"ignore_reinit_error": True}
	if args.ray_address and args.ray_address.lower() != "local":
		init_kwargs["address"] = args.ray_address

	ray.init(**init_kwargs)
	try:
		cluster = describe_ray_cluster()

		if args.target_node_ip:
			resource_key = f"node:{args.target_node_ip}"
			if resource_key not in ray.cluster_resources():
				raise RuntimeError(
					f"Requested target node resource {resource_key!r} not found in cluster resources"
				)

		actor_options = build_actor_options(args)
		actor = EnvSmokeActor.options(**actor_options).remote()
		result = cast(dict[str, Any], ray.get(actor.run.remote(dict(request))))
		result["ray_cluster"] = cluster
		result["ray_driver"] = {
			"ray_address": args.ray_address,
			"target_node_ip": args.target_node_ip,
			"actor_options": actor_options,
		}
		return result
	finally:
		ray.shutdown()


def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv)

	logging.basicConfig(
		level=getattr(logging, args.log_level),
		format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
	)

	request = build_smoke_request(args)
	if args.execution_mode == "local":
		if not args.grpc_host:
			raise SystemExit("--grpc-host is required in local mode so sidecars know where to dial back")

		result = run_env_smoke(request)
		result["ray_runtime"] = {
			"mode": "local",
			"detected_node_ip": get_node_ip_address(),
		}
	else:
		result = run_remote_smoke(args, request)

	print(json.dumps(json_safe(result), ensure_ascii=False, indent=2, sort_keys=True))
	return 0 if result.get("success") else 1


if __name__ == "__main__":
	raise SystemExit(main())
