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

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    # Fallback: build a minimal FastAPI app manually for local testing
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

    @app.get("/health")
    def health():
        return {"status": "ok", "env": "math_reasoning_env"}

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

    @app.post("/step")
    def step(action: dict):
        act = MathAction(
            reasoning=action.get("reasoning", ""),
            answer=action.get("answer", ""),
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

else:
    # Official OpenEnv path: use create_app helper
    from models import MathAction, MathObservation
    from server.math_environment import MathReasoningEnvironment

    app = create_app(
        MathReasoningEnvironment,
        MathAction,
        MathObservation,
        env_name="math_reasoning_env",
    )

    def main():
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

    if __name__ == "__main__":
        main()
