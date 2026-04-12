"""
Grader for easy_arithmetic task.
Single-step arithmetic: addition, subtraction, multiplication, division.
Returns a score in [0.0, 1.0] based on correctness and attempt efficiency.
"""

def grade(episode=None, *, observation=None, action=None, state=None, trajectory=None, **kwargs) -> float:
    """
    Score an episode step for the easy_arithmetic task.

    Args:
        episode:     Full episode passed by the OpenEnv Validator.
        observation: MathObservation object (or dict) from the environment.
        action:      MathAction that produced this observation (optional).
        state:       MathState at the time of grading (optional).
        trajectory:  Full episode trajectory list (optional).
        **kwargs:    Forward-compat catch-all.

    Returns:
        float in [0.0, 1.0] — higher is better.
    """
    if episode is not None and observation is None:
        if isinstance(episode, list) and len(episode) > 0:
            last_step = episode[-1]
            observation = last_step.get("observation", last_step) if isinstance(last_step, dict) else getattr(last_step, "observation", last_step)
        else:
            observation = episode.get("observation", episode) if isinstance(episode, dict) else getattr(episode, "observation", episode)

    # Accept both object-style and dict-style observations
    if isinstance(observation, dict):
        correct = observation.get("correct", False)
        reward  = float(observation.get("reward", 0.0))
        done    = observation.get("done", False)
    else:
        correct = getattr(observation, "correct", False)
        reward  = float(getattr(observation, "reward", 0.0))
        done    = getattr(observation, "done", False)

    # Normalize reward from env range [-0.5, 1.0] → [0.0, 1.0]
    score = (reward + 0.5) / 1.5

    # Hard cap: if episode ended without a correct answer, score is 0
    if done and not correct:
        score = 0.0

    return max(0.0, min(1.0, score))