"""
FastAPI application for the Math Reasoning Environment.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

    # Or run directly:
    python server/app.py
"""

import sys
import os

# Support both in-repo and standalone imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Shared helpers (used by both code paths below)
# ---------------------------------------------------------------------------

def _run_sample_grade(difficulty: str) -> float:
    """Run a sample grading to prove each grader returns a value in 0.0–1.0."""
    sample_obs = {
        "correct": True,
        "reward": 1.0,
        "done": True,
        "hint": "",
        "difficulty": difficulty,
    }
    if difficulty == "easy":
        from tasks.easy_arithmetic import grade
    elif difficulty == "medium":
        from tasks.medium_algebra import grade
    else:
        from tasks.hard_reasoning import grade
    return grade(observation=sample_obs)


def _tasks_payload() -> dict:
    """Build the /tasks response body."""
    return {
        "tasks": [
            {
                "name": "easy_arithmetic",
                "description": (
                    "Single-step arithmetic problems "
                    "(addition, subtraction, multiplication, division)"
                ),
                "difficulty": "easy",
                "max_steps": 3,
                "grader": "tasks.easy_arithmetic:grade",
                "sample_score": _run_sample_grade("easy"),
            },
            {
                "name": "medium_algebra",
                "description": (
                    "Multi-step algebra and word problems "
                    "(equations, speed-distance-time)"
                ),
                "difficulty": "medium",
                "max_steps": 3,
                "grader": "tasks.medium_algebra:grade",
                "sample_score": _run_sample_grade("medium"),
            },
            {
                "name": "hard_reasoning",
                "description": (
                    "Complex multi-step algebra and word problems "
                    "(quadratics, pipes, logarithms)"
                ),
                "difficulty": "hard",
                "max_steps": 3,
                "grader": "tasks.hard_reasoning:grade",
                "sample_score": _run_sample_grade("hard"),
            },
        ]
    }


# ---------------------------------------------------------------------------
# Code path A: fallback (no openenv-core installed)
# ---------------------------------------------------------------------------

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn

    from models import MathAction, MathObservation
    from server.math_environment import MathReasoningEnvironment

    env = MathReasoningEnvironment()
    app = FastAPI(
        title="Math Reasoning Environment",
        description=(
            "An OpenEnv-compatible RL environment for multi-step math "
            "reasoning across arithmetic, algebra, and word problems."
        ),
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ──────────────────────────────────────────────────────────────
    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "env": "math_reasoning_env",
            "version": "1.0.0",
            "protocol": "openenv-0.2.0",
        }

    # ── Tasks (required by OpenEnv validator) ───────────────────────────────
    @app.get("/tasks")
    def get_tasks():
        """Return all available tasks with grader info — required by OpenEnv validator."""
        return _tasks_payload()

    # ── Reset ───────────────────────────────────────────────────────────────
    @app.post("/reset")
    def reset(seed: int | None = None, difficulty: str | None = None):
        obs = env.reset(seed=seed, difficulty=difficulty)
        return {
            "observation": {
                "problem": obs.problem,
                "feedback": obs.feedback,
                "correct": obs.correct,
                "attempts_remaining": obs.attempts_remaining,
                "hint": obs.hint,
                "category": obs.category,
                "difficulty": obs.difficulty,
                "done": obs.done,
                "reward": obs.reward,
                "metadata": obs.metadata,
            }
        }

    # ── Step ────────────────────────────────────────────────────────────────
    @app.post("/step")
    def step(action: dict):
        # Handle both flat and nested payloads for robustness
        data = (
            action.get("action", action)
            if isinstance(action.get("action"), dict)
            else action
        )

        act = MathAction(
            reasoning=data.get("reasoning", ""),
            answer=data.get("answer", ""),
        )
        obs = env.step(act)
        return {
            "observation": {
                "problem": obs.problem,
                "feedback": obs.feedback,
                "correct": obs.correct,
                "attempts_remaining": obs.attempts_remaining,
                "hint": obs.hint,
                "category": obs.category,
                "difficulty": obs.difficulty,
                "done": obs.done,
                "reward": obs.reward,
                "metadata": obs.metadata,
            },
            "reward": obs.reward,
            "done": obs.done,
            "terminated": obs.done,
            "truncated": False,
            "info": obs.metadata,
        }

    # ── State ───────────────────────────────────────────────────────────────
    @app.get("/state")
    def state():
        s = env.state
        return {
            "episode_id": s.episode_id,
            "step_count": s.step_count,
            "problem": s.problem,
            "category": s.category,
            "difficulty": s.difficulty,
            "total_episodes": s.total_episodes,
            "correct_episodes": s.correct_episodes,
            "score": s.score,
        }

    def main():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    if __name__ == "__main__":
        main()

# ---------------------------------------------------------------------------
# Code path B: official OpenEnv path (openenv-core is installed)
# ---------------------------------------------------------------------------

else:
    from models import MathAction, MathObservation
    from server.math_environment import MathReasoningEnvironment

    env_instance = MathReasoningEnvironment()
    app = create_app(
        lambda: env_instance,
        MathAction,
        MathObservation,
        env_name="math_reasoning_env",
    )

    # ── Tasks (required by OpenEnv validator) ───────────────────────────────
    @app.get("/tasks")
    def get_tasks():
        """Return all available tasks with grader info — required by OpenEnv validator."""
        return _tasks_payload()

    # ── Global error handler ─────────────────────────────────────────────────
    from fastapi.responses import JSONResponse

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        import traceback
        return JSONResponse(
            status_code=500,
            content={"traceback": traceback.format_exc()},
        )

    def main():
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

    if __name__ == "__main__":
        main()