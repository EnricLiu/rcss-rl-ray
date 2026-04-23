from __future__ import annotations

import argparse
import json
import logging
import traceback
from collections.abc import Mapping, Sequence
from ipaddress import IPv4Address
from time import perf_counter
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


def _sample_action_for_agent(env: RCSSEnv, agent_id: int, payload: Any = None) -> dict[str, Any]:
	action_space = env.action_spaces[agent_id]
	sampled_action = action_space.sample()

	if not isinstance(payload, Mapping):
		return sampled_action

	act_mask = payload.get("act_mask")
	if not isinstance(act_mask, np.ndarray):
		return sampled_action

	mask = np.asarray(act_mask).astype(np.int8)
	allowed_indices = np.flatnonzero(mask)
	if allowed_indices.size == 0:
		logger.error("agent %s returned an empty act_mask; falling back to sampled discrete action", agent_id)
		return sampled_action

	sampled_action["actions"] = int(np.random.choice(allowed_indices))
	return sampled_action


def build_action_dict(
	env: RCSSEnv,
	active_agents: Sequence[int] | None = None,
	observations: Mapping[int, Any] | None = None,
) -> dict[int, Any]:
	agent_ids = list(active_agents) if active_agents is not None else list(env.agents)
	return {
		agent_id: _sample_action_for_agent(env, agent_id, None if observations is None else observations.get(agent_id))
		for agent_id in agent_ids
	}


def _safe_space_contains(space: Any, value: Any) -> bool | None:
	if space is None:
		return None
	try:
		return bool(space.contains(value))
	except Exception:
		return None


def _record_contract_issues(
	result: dict[str, Any],
	scope: str,
	issues: Sequence[str],
) -> None:
	if not issues:
		return

	result["contract_ok"] = False
	global_issues = cast(list[dict[str, Any]], result.setdefault("contract_issues", []))
	for issue in issues:
		entry = {"scope": scope, "issue": issue}
		if entry not in global_issues:
			global_issues.append(entry)
		logger.error("[contract:%s] %s", scope, issue)


def evaluate_reset_contract(
	env: RCSSEnv,
	obs: Mapping[int, Any],
	infos: Mapping[int, Any],
) -> dict[str, Any]:
	expected_agents = set(int(agent_id) for agent_id in env.agents)
	obs_agents = set(int(agent_id) for agent_id in obs.keys())
	info_agents = set(int(agent_id) for agent_id in infos.keys())
	issues: list[str] = []

	if not obs_agents:
		issues.append("reset returned no observations")
	if obs_agents != expected_agents:
		issues.append(
			f"reset obs agents mismatch: expected={sorted(expected_agents)}, got={sorted(obs_agents)}"
		)
	if info_agents != expected_agents:
		issues.append(
			f"reset info agents mismatch: expected={sorted(expected_agents)}, got={sorted(info_agents)}"
		)

	space_mismatches: list[int] = []
	unknown_space_checks: list[int] = []
	for agent_id, agent_obs in obs.items():
		contains = _safe_space_contains(env.observation_spaces.get(agent_id), agent_obs)
		if contains is False:
			space_mismatches.append(int(agent_id))
		elif contains is None:
			unknown_space_checks.append(int(agent_id))

	if space_mismatches:
		issues.append(
			f"reset observation payload does not match declared observation_space for agents={sorted(space_mismatches)}"
		)
	if unknown_space_checks:
		issues.append(
			f"reset observation_space.contains could not be evaluated for agents={sorted(unknown_space_checks)}"
		)

	return {
		"ok": not issues,
		"expected_agents": sorted(expected_agents),
		"obs_agents": sorted(obs_agents),
		"info_agents": sorted(info_agents),
		"issues": issues,
	}


def evaluate_step_contract(
	env: RCSSEnv,
	obs: Mapping[int, Any],
	rewards: Mapping[int, Any],
	terminateds: Mapping[Any, Any],
	truncateds: Mapping[Any, Any],
	infos: Mapping[int, Any],
) -> dict[str, Any]:
	expected_agents = set(int(agent_id) for agent_id in env.agents)
	obs_agents = set(int(agent_id) for agent_id in obs.keys())
	reward_agents = set(int(agent_id) for agent_id in rewards.keys())
	info_agents = set(int(agent_id) for agent_id in infos.keys())
	issues: list[str] = []

	if "__all__" not in terminateds:
		issues.append("terminateds is missing __all__")
	if "__all__" not in truncateds:
		issues.append("truncateds is missing __all__")
	if obs_agents != expected_agents:
		issues.append(
			f"step obs agents mismatch: expected={sorted(expected_agents)}, got={sorted(obs_agents)}"
		)
	if reward_agents != expected_agents:
		issues.append(
			f"step reward agents mismatch: expected={sorted(expected_agents)}, got={sorted(reward_agents)}"
		)
	if info_agents != expected_agents:
		issues.append(
			f"step info agents mismatch: expected={sorted(expected_agents)}, got={sorted(info_agents)}"
		)

	space_mismatches: list[int] = []
	unknown_space_checks: list[int] = []
	for agent_id, agent_obs in obs.items():
		contains = _safe_space_contains(env.observation_spaces.get(agent_id), agent_obs)
		if contains is False:
			space_mismatches.append(int(agent_id))
		elif contains is None:
			unknown_space_checks.append(int(agent_id))

	if space_mismatches:
		issues.append(
			f"step observation payload does not match declared observation_space for agents={sorted(space_mismatches)}"
		)
	if unknown_space_checks:
		issues.append(
			f"step observation_space.contains could not be evaluated for agents={sorted(unknown_space_checks)}"
		)

	return {
		"ok": not issues,
		"expected_agents": sorted(expected_agents),
		"obs_agents": sorted(obs_agents),
		"reward_agents": sorted(reward_agents),
		"info_agents": sorted(info_agents),
		"terminated_all": bool(terminateds.get("__all__")),
		"truncated_all": bool(truncateds.get("__all__")),
		"issues": issues,
	}


def determine_done_reason(
	terminateds: Mapping[Any, Any],
	truncateds: Mapping[Any, Any],
	steps_completed: int,
	steps_requested: int,
) -> str:
	if terminateds.get("__all__"):
		return "terminated"
	if truncateds.get("__all__"):
		return "truncated"
	if steps_completed >= steps_requested:
		return "step_limit_reached"
	return "incomplete"


def extract_scoreboard(infos: Mapping[int, Any]) -> dict[str, Any] | None:
	for info in infos.values():
		if isinstance(info, Mapping) and isinstance(info.get("scores"), Mapping):
			return json_safe(info["scores"])
	return None


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
		"episodes": args.episodes,
		"step_log_interval": args.step_log_interval,
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
	episodes = int(request.get("episodes", 1))
	step_log_interval = max(1, int(request.get("step_log_interval", 1)))

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
		"contract_ok": True,
		"contract_issues": [],
		"grpc_host": grpc_host,
		"grpc_port": grpc_port,
		"allocator_base_url": env.config.allocator.base_url,
		"agent_unums": list(env.agents),
		"steps_per_episode_requested": steps,
		"episodes_requested": episodes,
		"schema_summary": {
			"agent_team": env.schema.teams.agent_team.name,
			"left_team": env.schema.teams.left.name,
			"right_team": env.schema.teams.right.name,
			"time_up": env.schema.stopping.time_up,
		},
		"totals": {
			"episodes_completed": 0,
			"steps_completed": 0,
			"terminated_episodes": 0,
			"truncated_episodes": 0,
			"step_limit_episodes": 0,
			"reward_sum": 0.0,
		},
		"episodes": [],
	}
	current_phase = "bootstrap"
	current_episode_idx: int | None = None
	current_step_idx: int | None = None
	current_episode_record: dict[str, Any] | None = None

	try:
		logger.warning(
			"starting env durability smoke: grpc_host=%s grpc_port=%d episodes=%d steps_per_episode=%d num_agents=%d",
			grpc_host,
			grpc_port,
			episodes,
			steps,
			num_agents,
		)
		current_phase = "allocator_probe"
		allocator_version = env.allocator.fleet_get_template_version()
		result["allocator_template_version"] = allocator_version
		logger.warning("allocator template version: %s", allocator_version)
		result["allocator_health"] = env.allocator.health_check()
		result["allocator_ready"] = env.allocator.readiness_check()
		logger.warning(
			"allocator readiness: health=%s ready=%s",
			result["allocator_health"],
			result["allocator_ready"],
		)

		for episode_idx in range(1, episodes + 1):
			current_episode_idx = episode_idx
			current_step_idx = None
			current_episode_record = {
				"episode_index": episode_idx,
				"status": "running",
				"steps_requested": steps,
				"step_log_interval": step_log_interval,
			}
			logger.warning("episode %d/%d: reset start", episode_idx, episodes)

			current_phase = "reset"
			reset_started_at = perf_counter()
			obs, infos = env.reset()
			reset_duration_s = perf_counter() - reset_started_at
			current_episode_record["reset_duration_s"] = round(reset_duration_s, 6)
			current_episode_record["reset"] = {
				"obs_agents": sorted(obs.keys()),
				"obs_summary": summarize_observations(obs),
				"infos": json_safe(infos),
			}

			reset_contract = evaluate_reset_contract(env, obs, infos)
			current_episode_record["reset_contract"] = reset_contract
			_record_contract_issues(result, f"episode_{episode_idx}.reset", reset_contract["issues"])

			logger.warning(
				"episode %d/%d: reset done in %.3fs obs_agents=%s",
				episode_idx,
				episodes,
				reset_duration_s,
				sorted(obs.keys()),
			)

			current_phase = "post_reset_diagnostics"
			current_episode_record["post_reset_diagnostics"] = collect_room_diagnostics(env)
			logger.warning(
				"episode %d/%d: room=%s rcss=%s mc=%s",
				episode_idx,
				episodes,
				env.room.info.name,
				env.room.info.base_url_rcss,
				env.room.info.base_url_mc,
			)

			step_records: list[dict[str, Any]] = []
			step_latencies_s: list[float] = []
			total_reward_sum = 0.0
			active_agents: list[int] = [int(agent_id) for agent_id in sorted(obs.keys())] if obs else [int(agent_id) for agent_id in env.agents]
			done_reason = "step_limit_reached"
			last_infos: Mapping[int, Any] = infos

			for step_idx in range(steps):
				current_phase = "step"
				current_step_idx = step_idx + 1
				actions = build_action_dict(env, active_agents, obs)
				step_started_at = perf_counter()
				next_obs, rewards, terminateds, truncateds, next_infos = env.step(actions)
				step_duration_s = perf_counter() - step_started_at
				step_latencies_s.append(step_duration_s)
				step_reward_sum = float(sum(rewards.values()))
				total_reward_sum += step_reward_sum
				last_infos = next_infos

				step_contract = evaluate_step_contract(env, next_obs, rewards, terminateds, truncateds, next_infos)
				_record_contract_issues(result, f"episode_{episode_idx}.step_{step_idx + 1}", step_contract["issues"])

				step_record = {
					"step_index": step_idx + 1,
					"duration_s": round(step_duration_s, 6),
					"action_agents": sorted(actions.keys()),
					"reward_sum": step_reward_sum,
					"rewards": json_safe(rewards),
					"terminateds": json_safe(terminateds),
					"truncateds": json_safe(truncateds),
					"infos": json_safe(next_infos),
					"obs_summary": summarize_observations(next_obs),
					"contract": step_contract,
				}
				step_records.append(step_record)

				should_log_step = (
					(step_idx + 1) == 1
					or (step_idx + 1) % step_log_interval == 0
					or bool(terminateds.get("__all__"))
					or bool(truncateds.get("__all__"))
				)
				if should_log_step:
					logger.warning(
						"episode %d/%d step %d/%d: duration=%.3fs reward_sum=%.3f terminated=%s truncated=%s obs_agents=%s",
						episode_idx,
						episodes,
						step_idx + 1,
						steps,
						step_duration_s,
						step_reward_sum,
						bool(terminateds.get("__all__")),
						bool(truncateds.get("__all__")),
						sorted(next_obs.keys()),
					)

				obs = next_obs
				active_agents = [int(agent_id) for agent_id in sorted(next_obs.keys())] if next_obs else active_agents
				done_reason = determine_done_reason(terminateds, truncateds, step_idx + 1, steps)

				if terminateds.get("__all__") or truncateds.get("__all__"):
					break

			current_step_idx = None
			steps_completed = len(step_records)
			current_episode_record["steps"] = step_records
			current_episode_record["steps_completed"] = steps_completed
			current_episode_record["reward_sum_total"] = total_reward_sum
			current_episode_record["done_reason"] = done_reason
			current_episode_record["final_scores"] = extract_scoreboard(last_infos)
			current_episode_record["step_latency_s"] = {
				"min": round(min(step_latencies_s), 6) if step_latencies_s else None,
				"max": round(max(step_latencies_s), 6) if step_latencies_s else None,
				"avg": round(sum(step_latencies_s) / len(step_latencies_s), 6) if step_latencies_s else None,
			}
			current_episode_record["status"] = "success"

			if done_reason == "terminated":
				result["totals"]["terminated_episodes"] += 1
			elif done_reason == "truncated":
				result["totals"]["truncated_episodes"] += 1
			else:
				result["totals"]["step_limit_episodes"] += 1

			result["totals"]["episodes_completed"] += 1
			result["totals"]["steps_completed"] += steps_completed
			result["totals"]["reward_sum"] += total_reward_sum
			completed_episode_record = current_episode_record
			cast(list[dict[str, Any]], result["episodes"]).append(completed_episode_record)
			logger.warning(
				"episode %d/%d finished: steps_completed=%d done_reason=%s reward_sum=%.3f final_scores=%s",
				episode_idx,
				episodes,
				steps_completed,
				done_reason,
				total_reward_sum,
				completed_episode_record["final_scores"],
			)
			current_episode_record = None

		result["success"] = True
		logger.warning(
			"env durability smoke finished successfully: episodes_completed=%d steps_completed=%d contract_ok=%s",
			result["totals"]["episodes_completed"],
			result["totals"]["steps_completed"],
			result["contract_ok"],
		)
		return result
	except Exception as exc:  # pragma: no cover - exercised against live cluster
		logger.error(
			"env durability smoke failed: phase=%s episode=%s step=%s error=%s: %s",
			current_phase,
			current_episode_idx,
			current_step_idx,
			type(exc).__name__,
			str(exc),
		)
		logger.exception("env durability smoke traceback")
		result["error"] = {
			"type": type(exc).__name__,
			"message": str(exc),
			"traceback": traceback.format_exc(),
		}
		result["failure_context"] = {
			"phase": current_phase,
			"episode_index": current_episode_idx,
			"step_index": current_step_idx,
		}
		if current_episode_record is not None:
			current_episode_record["status"] = "failed"
			current_episode_record["failure_context"] = result["failure_context"]
			failed_episode_record = current_episode_record
			cast(list[dict[str, Any]], result["episodes"]).append(failed_episode_record)
		if env.has_room():
			try:
				result["failure_room"] = json_safe(env.room.info)
			except Exception:
				pass
		return result
	finally:
		try:
			logger.warning("closing env smoke resources")
			env.close()
		except Exception as exc:  # pragma: no cover - exercised against live cluster
			logger.error("failed to close env smoke resources: %s: %s", type(exc).__name__, str(exc))
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

	parser.add_argument("--num-agents", type=int, default=2)
	parser.add_argument("--episodes", type=int, default=100)
	parser.add_argument("--steps", type=int, default=6000)
	parser.add_argument(
		"--step-log-interval",
		type=int,
		default=1,
		help="Emit warning logs every N steps within each episode",
	)
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
