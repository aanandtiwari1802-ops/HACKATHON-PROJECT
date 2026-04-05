"""
Sample Interface Script — Math Reasoning Environment.

This script demonstrates how to run the web interface for the
Math Reasoning Environment using OpenEnv's built-in web UI.

Requirements:
    pip install openenv-core fastapi uvicorn

Usage:
    # Enable web interface and run:
    ENABLE_WEB_INTERFACE=true python sample_interface.py

    # Then open in browser:
    # http://localhost:8000/web

The web interface provides:
  - Left pane:  Submit reasoning + answer (HumanAgent interaction)
  - Right pane: Live observation state (problem, feedback, score)
  - Action log: Full history of all attempts and rewards
"""

import os
import sys

# Add environment to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Attempt to use OpenEnv's create_web_interface_app ──────────────────────
try:
    from openenv.core.env_server import create_web_interface_app
    from models import MathAction, MathObservation
    from server.math_environment import MathReasoningEnvironment

    # 1. Instantiate the environment
    env = MathReasoningEnvironment()

    # 2. Create the web interface app
    #    This creates a two-pane layout:
    #      - Left:  action form (reasoning textarea + answer input)
    #      - Right: current observation state (problem, feedback, score)
    app = create_web_interface_app(env, MathAction, MathObservation)

    print("=" * 60)
    print("  Math Reasoning Environment — Web Interface")
    print("=" * 60)
    print()
    print("  ✅ Web interface created successfully.")
    print()
    print("  Run with:")
    print("    ENABLE_WEB_INTERFACE=true python sample_interface.py")
    print()
    print("  Then visit:  http://localhost:8000/web")
    print("=" * 60)

    if __name__ == "__main__":
        import uvicorn
        os.environ["ENABLE_WEB_INTERFACE"] = "true"
        uvicorn.run(app, host="0.0.0.0", port=8000)

except ImportError:
    # ── Fallback: start the server and show interaction demo ──────────────
    print("=" * 60)
    print("  Math Reasoning Environment — Demo Interface")
    print("  (openenv-core not installed — using standalone mode)")
    print("=" * 60)
    print()

    from server.math_environment import MathReasoningEnvironment
    from models import MathAction

    env = MathReasoningEnvironment(seed=42)

    print("📚 Starting a new episode...\n")
    obs = env.reset(difficulty="medium")
    print(f"  Problem   : {obs.problem}")
    print(f"  Difficulty: {obs.difficulty}")
    print(f"  Attempts  : {obs.attempts_remaining}")
    print()

    # Simulated agent attempts
    attempts = [
        MathAction(
            reasoning="Speed = Distance / Time = 240 / 3 = 80 km/h",
            answer="80",
        ),
    ]

    for i, action in enumerate(attempts, 1):
        print(f"  Attempt {i}:")
        print(f"    Reasoning : {action.reasoning}")
        print(f"    Answer    : {action.answer}")
        result = env.step(action)
        print(f"    Feedback  : {result.feedback}")
        print(f"    Reward    : {result.reward}")
        print(f"    Done      : {result.done}")
        print()
        if result.done:
            break

    state = env.state
    print(f"  Episode score: {state.correct_episodes}/{state.total_episodes}")
    print()
    print("  To run the full HTTP server:")
    print("    cd math_reasoning_env")
    print("    uvicorn server.app:app --host 0.0.0.0 --port 8000")
    print("=" * 60)
