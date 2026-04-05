"""
Models for the Math Reasoning Environment.

Defines type-safe Action, Observation, and State classes
using Pydantic as required by OpenEnv.
"""

from typing import Any, Dict, List, Optional
try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    from dataclasses import dataclass, field
    HAS_PYDANTIC = False

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Fallback for standalone development/testing without openenv-core
    if HAS_PYDANTIC:
        class Action(BaseModel):
            metadata: Dict[str, Any] = Field(default_factory=dict)
        class Observation(BaseModel):
            done: bool = False
            reward: Optional[float] = None
            metadata: Dict[str, Any] = Field(default_factory=dict)
        class State(BaseModel):
            episode_id: Optional[str] = None
            step_count: int = 0
    else:
        @dataclass
        class Action:
            metadata: Dict[str, Any] = field(default_factory=dict)
        @dataclass
        class Observation:
            done: bool = False
            reward: Optional[float] = None
            metadata: Dict[str, Any] = field(default_factory=dict)
        @dataclass
        class State:
            episode_id: Optional[str] = None
            step_count: int = 0


# If we are using Pydantic, we DO NOT use @dataclass decorator.
# Inheriting from Action/Observation/State (which are BaseModels) is enough.

if HAS_PYDANTIC:
    class MathAction(Action):
        reasoning: str = ""
        answer: str = ""
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class MathObservation(Observation):
        problem: str = ""
        feedback: str = ""
        correct: bool = False
        attempts_remaining: int = 3
        hint: str = ""
        category: str = ""
        difficulty: str = "medium"
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class MathState(State):
        episode_id: Optional[str] = None
        step_count: int = 0
        correct_answer: str = ""
        problem: str = ""
        category: str = ""
        difficulty: str = "medium"
        total_episodes: int = 0
        correct_episodes: int = 0
        score: float = 0.0
else:
    @dataclass
    class MathAction(Action):
        reasoning: str = ""
        answer: str = ""
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class MathObservation(Observation):
        problem: str = ""
        feedback: str = ""
        correct: bool = False
        attempts_remaining: int = 3
        hint: str = ""
        category: str = ""
        difficulty: str = "medium"
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class MathState(State):
        episode_id: Optional[str] = None
        step_count: int = 0
        correct_answer: str = ""
        problem: str = ""
        category: str = ""
        difficulty: str = "medium"
        total_episodes: int = 0
        correct_episodes: int = 0
        score: float = 0.0
