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

    ball_to_goal_distance = curr_obs.ball.position

    return rewards


def distance(pos1: pb2.Vector2D, pos2: pb2.Vector2D) -> float:
    return ((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2) ** 0.5

