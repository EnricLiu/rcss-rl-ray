from grpc_srv import pb2

def calculate(
    prev_obs: pb2.WorldModel | None,
    prev_truth: pb2.WorldModel | None,
    curr_obs: pb2.WorldModel,
    curr_truth: pb2.WorldModel,
) -> float:
    rewards = 0.0

    # Goal reward (+1 for our goal, -1 for opponent goal).
    score_diff = curr_obs.our_team_score - prev_obs.our_team_score
    opp_diff = curr_obs.their_team_score - prev_obs.their_team_score
    rewards += float(score_diff)
    rewards -= float(opp_diff)

    # Small distance-to-ball penalty to encourage engagement.
    rewards -= 0.001 * curr_obs.self.dist_from_ball

    return rewards
