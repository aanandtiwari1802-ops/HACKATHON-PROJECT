"""
Models for the Math Reasoning Environment.

Defines type-safe Action, Observation, and State classes
using dataclasses as required by OpenEnv.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Fallback for standalone development/testing
    from dataclasses import dataclass as _dc

    @_dc
    class Action:
        metadata: Dict[str, Any] = field(default_factory=dict)

    @_dc
    class Observation:
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = field(default_factory=dict)

    @_dc
    class State:
        episode_id: Optional[str] = None
        step_count: int = 0


@dataclass
class MathAction(Action):
    """
    Action submitted by an RL agent to the Math Reasoning Environment.

    The agent provides:
    - reasoning: a chain-of-thought explanation of their work
    - answer: the final numeric answer as a string (e.g. "42" or "3.14")
    """

    reasoning: str = ""      # Agent's step-by-step working
    answer: str = ""         # Final answer (numeric string)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MathObservation(Observation):
    """
    Observation returned by the Math Reasoning Environment after each step.

    Fields:
    - problem: the current problem statement shown to the agent
    - feedback: feedback on the previous attempt (empty on first reset)
    - correct: whether the last answer was correct
    - attempts_remaining: how many guesses are left this episode
    - hint: optional hint shown after a wrong answer
    - done: whether the episode has ended
    - reward: reward received for the last action
    - metadata: additional info dict
    """

    problem: str = ""
    feedback: str = ""
    correct: bool = False
    attempts_remaining: int = 3
    hint: str = ""
    category: str = ""          # e.g. "arithmetic", "algebra", "word_problem"
    difficulty: str = "medium"  # "easy", "medium", "hard"
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MathState(State):
    """
    Episode state for the Math Reasoning Environment.
    """

    episode_id: Optional[str] = None
    step_count: int = 0
    correct_answer: str = ""        # Ground-truth (hidden from agent)
    problem: str = ""
    category: str = ""
    difficulty: str = "medium"
    total_episodes: int = 0
    correct_episodes: int = 0
    score: float = 0.0              # Running accuracy
