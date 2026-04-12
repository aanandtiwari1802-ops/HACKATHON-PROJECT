"""
Grader for medium_algebra tasks.

Signature: grade(episode: dict) -> float
Score must be strictly between 0.0 and 1.0 (exclusive).
"""
from __future__ import annotations


def grade(episode: dict) -> float:
    success: bool    = bool(episode.get("success", False))
    steps: int       = int(episode.get("steps", 1))
    rewards: list    = episode.get("rewards", [])
    env_score: float = float(episode.get("score", 0.0))

    if success:
        if steps <= 1:
            return 0.99   # perfect — but strictly < 1.0
        elif steps == 2:
            return 0.55
        else:
            return 0.25

    if rewards:
        avg = sum(rewards) / len(rewards)
        return round(max(0.01, min(0.14, avg)), 4)

    return round(max(0.01, min(0.14, env_score if env_score > 0 else 0.01)), 4)