"""
Grader for the easy_arithmetic task.

The grader receives the episode result and returns a score
strictly between 0 and 1 (exclusive), as required by the validator.
"""


def grade(episode_result: dict) -> float:
    """
    Score an easy_arithmetic episode.

    Args:
        episode_result: dict with keys such as 'reward', 'correct',
                        'attempts', 'score', produced by the environment.

    Returns:
        float strictly in (0, 1).
    """
    raw = float(episode_result.get("score", episode_result.get("reward", 0.5)))
    # Clamp strictly inside (0, 1) — validator rejects 0.0 and 1.0
    return max(0.01, min(raw, 0.99))
