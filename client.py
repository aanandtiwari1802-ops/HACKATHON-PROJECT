"""
Client for the Math Reasoning Environment.

Extends OpenEnv's HTTPEnvClient (or provides a standalone fallback)
for easy integration with RL training loops.

Usage:
    # Sync (simple training loop):
    from math_reasoning_env import MathReasoningEnv, MathAction

    with MathReasoningEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset()
        print(result.observation.problem)

        result = env.step(MathAction(
            reasoning="47 + 38: 40+30=70, 7+8=15, total=85",
            answer="85"
        ))
        print(result.observation.feedback)
        print(result.reward)

    # Async (recommended for production):
    import asyncio
    from math_reasoning_env import MathReasoningEnv, MathAction

    async def main():
        async with MathReasoningEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            result = await env.step(MathAction(reasoning="...", answer="85"))
            print(result.observation.feedback)

    asyncio.run(main())
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from openenv.core.env_client import EnvClient, StepResult
    from .models import MathAction, MathObservation, MathState

    class MathReasoningEnv(EnvClient[MathAction, MathObservation, MathState]):
        """
        HTTP client for the Math Reasoning Environment.

        Connects to a running math_reasoning_env server and exposes
        reset() / step() / state() with full type safety.
        """

        def _step_payload(self, action: MathAction) -> dict[str, Any]:
            """Serialize MathAction to JSON payload."""
            return {
                "reasoning": action.reasoning,
                "answer": action.answer,
                "metadata": action.metadata,
            }

        def _parse_result(self, payload: dict) -> StepResult[MathObservation]:
            """Deserialize HTTP response to StepResult[MathObservation]."""
            obs_data = payload.get("observation", payload)
            obs = MathObservation(
                problem=obs_data.get("problem", ""),
                feedback=obs_data.get("feedback", ""),
                correct=obs_data.get("correct", False),
                attempts_remaining=obs_data.get("attempts_remaining", 3),
                hint=obs_data.get("hint", ""),
                category=obs_data.get("category", ""),
                difficulty=obs_data.get("difficulty", "medium"),
                done=obs_data.get("done", False),
                reward=obs_data.get("reward"),
                metadata=obs_data.get("metadata", {}),
            )
            return StepResult(
                observation=obs,
                reward=payload.get("reward", obs.reward),
                done=payload.get("done", obs.done),
            )

        def _parse_state(self, payload: dict) -> MathState:
            """Deserialize HTTP response to MathState."""
            return MathState(
                episode_id=payload.get("episode_id"),
                step_count=payload.get("step_count", 0),
                correct_answer=payload.get("correct_answer", ""),
                problem=payload.get("problem", ""),
                category=payload.get("category", ""),
                difficulty=payload.get("difficulty", "medium"),
                total_episodes=payload.get("total_episodes", 0),
                correct_episodes=payload.get("correct_episodes", 0),
                score=payload.get("score", 0.0),
            )

except ImportError:
    # Standalone fallback: pure requests-based sync client
    import requests
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models import MathAction, MathObservation, MathState

    class _StepResult:
        def __init__(self, observation, reward, done):
            self.observation = observation
            self.reward = reward
            self.done = done

    class MathReasoningEnv:
        """
        Standalone sync HTTP client for Math Reasoning Environment.
        Works without openenv-core installed.
        """

        def __init__(self, base_url: str = "http://localhost:8000"):
            self.base_url = base_url.rstrip("/")
            self._session = requests.Session()

        # Context manager support
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._session.close()

        def sync(self):
            """Return self for sync() chaining compatibility."""
            return self

        def reset(
            self,
            seed: Optional[int] = None,
            difficulty: Optional[str] = None,
        ) -> _StepResult:
            params = {}
            if seed is not None:
                params["seed"] = seed
            if difficulty is not None:
                params["difficulty"] = difficulty
            r = self._session.post(f"{self.base_url}/reset", params=params)
            r.raise_for_status()
            return self._parse_response(r.json())

        def step(self, action: MathAction) -> _StepResult:
            payload = {
                "reasoning": action.reasoning,
                "answer": action.answer,
            }
            r = self._session.post(f"{self.base_url}/step", json=payload)
            r.raise_for_status()
            return self._parse_response(r.json())

        def state(self) -> MathState:
            r = self._session.get(f"{self.base_url}/state")
            r.raise_for_status()
            data = r.json()
            return MathState(
                episode_id=data.get("episode_id"),
                step_count=data.get("step_count", 0),
                problem=data.get("problem", ""),
                category=data.get("category", ""),
                difficulty=data.get("difficulty", "medium"),
                total_episodes=data.get("total_episodes", 0),
                correct_episodes=data.get("correct_episodes", 0),
                score=data.get("score", 0.0),
            )

        def close(self):
            self._session.close()

        def _parse_response(self, data: dict) -> _StepResult:
            obs_data = data.get("observation", data)
            obs = MathObservation(
                problem=obs_data.get("problem", ""),
                feedback=obs_data.get("feedback", ""),
                correct=obs_data.get("correct", False),
                attempts_remaining=obs_data.get("attempts_remaining", 3),
                hint=obs_data.get("hint", ""),
                category=obs_data.get("category", ""),
                difficulty=obs_data.get("difficulty", "medium"),
                done=obs_data.get("done", False),
                reward=obs_data.get("reward"),
                metadata=obs_data.get("metadata", {}),
            )
            return _StepResult(
                observation=obs,
                reward=data.get("reward", obs.reward),
                done=data.get("done", obs.done),
            )
