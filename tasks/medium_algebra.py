"""
Grader for medium_algebra task.
Multi-step algebra and word problems (equations, speed-distance-time).
Penalizes use of hints (shown on 3rd wrong attempt) slightly.
"""

def grade(*, observation, action=None, state=None, trajectory=None, **kwargs) -> float:
    """
    Score an episode step for the medium_algebra task.

    Returns:
        float in [0.0, 1.0].
    """
    if isinstance(observation, dict):
        correct           = observation.get("correct", False)
        reward            = float(observation.get("reward", 0.0))
        done              = observation.get("done", False)
        hint_was_shown    = bool(observation.get("hint", ""))
    else:
        correct           = getattr(observation, "correct", False)
        reward            = float(getattr(observation, "reward", 0.0))
        done              = getattr(observation, "done", False)
        hint_was_shown    = bool(getattr(observation, "hint", ""))

    # Normalize reward [-0.5, 1.0] → [0.0, 1.0]
    score = (reward + 0.5) / 1.5

    # Apply a small penalty if the agent needed a hint
    if correct and hint_was_shown:
        score = max(0.0, score - 0.1)

    if done and not correct:
        score = 0.0

    return max(0.0, min(1.0, score))