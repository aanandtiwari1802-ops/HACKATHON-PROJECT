"""
Grader for hard_reasoning task.
Complex multi-step algebra and word problems (quadratics, pipes, logarithms).
Rewards clean chain-of-thought reasoning in addition to correctness.
"""

_MIN_REASONING_WORDS = 10  # Expect substantive reasoning for hard problems

def grade(episode=None, *, observation=None, action=None, state=None, trajectory=None, **kwargs) -> float:
    """
    Score an episode step for the hard_reasoning task.

    Adds a small bonus (up to 0.1) when the agent produces a detailed
    chain-of-thought reasoning string, as expected for hard problems.

    Returns:
        float in [0.0, 1.0].
    """
    if episode is not None and observation is None:
        if isinstance(episode, list) and len(episode) > 0:
            last_step = episode[-1]
            observation = last_step.get("observation", last_step) if isinstance(last_step, dict) else getattr(last_step, "observation", last_step)
            if action is None:
                action = last_step.get("action", None) if isinstance(last_step, dict) else getattr(last_step, "action", None)
        else:
            observation = episode.get("observation", episode) if isinstance(episode, dict) else getattr(episode, "observation", episode)
            if action is None:
                action = episode.get("action", None) if isinstance(episode, dict) else getattr(episode, "action", None)

    if isinstance(observation, dict):
        correct        = observation.get("correct", False)
        reward         = float(observation.get("reward", 0.0))
        done           = observation.get("done", False)
    else:
        correct        = getattr(observation, "correct", False)
        reward         = float(getattr(observation, "reward", 0.0))
        done           = getattr(observation, "done", False)

    # Normalize reward [-0.5, 1.0] → [0.0, 1.0]
    score = (reward + 0.5) / 1.5

    if done and not correct:
        return 0.0

    # Reasoning quality bonus
    if action is not None:
        reasoning = ""
        if isinstance(action, dict):
            reasoning = action.get("reasoning", "")
        else:
            reasoning = getattr(action, "reasoning", "")
        word_count = len(reasoning.split())
        if correct and word_count >= _MIN_REASONING_WORDS:
            # Bonus scales up to 0.1 for 30+ word reasoning chains
            bonus = min(0.1, word_count / 300)
            score = min(1.0, score + bonus)

    return max(0.0, min(1.0, score))