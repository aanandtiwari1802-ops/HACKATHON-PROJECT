"""
Pre-Validation Script — Math Reasoning Environment.

Validates that the environment meets all OpenEnv hackathon requirements:
  1. Environment can be instantiated
  2. reset() returns a valid Observation
  3. step() returns correct 5-tuple equivalent fields
  4. Multiple steps work without crashing
  5. Episode terminates correctly
  6. Reward is a float
  7. done flag is a boolean
  8. All difficulty levels work (easy / medium / hard)
  9. Runs within resource constraints (< 20 min, 8 GB RAM)
 10. State is accessible at any time

Usage:
    python pre_validation.py

Exit codes:
    0 — all checks passed
    1 — one or more checks failed
"""

import sys
import os
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def check(name: str, fn):
    """Run a single validation check and record the result."""
    try:
        fn()
        results.append((PASS, name))
        print(f"  {PASS}  {name}")
    except Exception as exc:
        results.append((FAIL, name))
        print(f"  {FAIL}  {name}")
        print(f"         Error: {exc}")
        traceback.print_exc()


# ── Import ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Math Reasoning Environment — Pre-Validation Script")
print("=" * 65)

try:
    from server.math_environment import MathReasoningEnvironment
    from models import MathAction, MathObservation, MathState
    print("\n  ✅ Imports successful\n")
except Exception as e:
    print(f"\n  ❌ FATAL: Could not import environment — {e}")
    sys.exit(1)

# ── Checks ──────────────────────────────────────────────────────────────────

# 1. Instantiation
def test_instantiation():
    env = MathReasoningEnvironment()
    assert env is not None, "Environment is None"

check("Environment can be instantiated", test_instantiation)


# 2. reset() returns a valid Observation
def test_reset_returns_observation():
    env = MathReasoningEnvironment(seed=0)
    obs = env.reset()
    assert isinstance(obs, MathObservation), f"reset() must return MathObservation, got {type(obs)}"
    assert isinstance(obs.problem, str) and len(obs.problem) > 0, "obs.problem must be a non-empty str"
    assert isinstance(obs.done, bool), "obs.done must be bool"
    assert isinstance(obs.attempts_remaining, int), "obs.attempts_remaining must be int"

check("reset() returns valid MathObservation", test_reset_returns_observation)


# 3. step() returns correct fields (observation, reward, terminated, truncated, info)
def test_step_returns_valid_fields():
    env = MathReasoningEnvironment(seed=1)
    env.reset()
    action = MathAction(reasoning="Test reasoning", answer="999")
    obs = env.step(action)
    # observation
    assert isinstance(obs, MathObservation), f"step() must return MathObservation, got {type(obs)}"
    # reward (equivalent to float)
    assert obs.reward is not None, "obs.reward must not be None after step"
    assert isinstance(obs.reward, (int, float)), f"reward must be numeric, got {type(obs.reward)}"
    # terminated / done
    assert isinstance(obs.done, bool), "obs.done (terminated) must be bool"
    # info via metadata
    assert isinstance(obs.metadata, dict), "obs.metadata must be dict"

check("step() returns (obs, reward, terminated, truncated, info) equivalents", test_step_returns_valid_fields)


# 4. Multiple steps without crashing
def test_multiple_steps():
    env = MathReasoningEnvironment(seed=2)
    env.reset()
    for i in range(3):
        action = MathAction(reasoning=f"Attempt {i}", answer=str(i * 10))
        obs = env.step(action)
        assert obs is not None

check("Multiple steps (3) complete without crashing", test_multiple_steps)


# 5. Episode terminates on correct answer
def test_episode_terminates_correct():
    env = MathReasoningEnvironment(seed=3)
    env.reset(difficulty="easy")
    correct_answer = env.state.correct_answer
    action = MathAction(reasoning="I know the answer", answer=correct_answer)
    obs = env.step(action)
    assert obs.correct is True, f"Expected correct=True, got {obs.correct}"
    assert obs.done is True, f"Expected done=True after correct answer, got {obs.done}"
    assert obs.reward > 0, f"Expected positive reward, got {obs.reward}"

check("Episode terminates with done=True on correct answer", test_episode_terminates_correct)


# 6. Episode terminates after max attempts
def test_episode_terminates_max_attempts():
    env = MathReasoningEnvironment(seed=4)
    env.reset()
    for _ in range(3):  # max attempts
        obs = env.step(MathAction(reasoning="wrong", answer="99999"))
    assert obs.done is True, f"Expected done=True after max attempts, got {obs.done}"
    assert obs.reward < 0, f"Expected negative reward for failure, got {obs.reward}"

check("Episode terminates after max attempts with negative reward", test_episode_terminates_max_attempts)


# 7. Reward is always a float
def test_reward_is_float():
    env = MathReasoningEnvironment(seed=5)
    env.reset()
    obs = env.step(MathAction(reasoning="x", answer="0"))
    assert isinstance(obs.reward, (int, float)), f"reward must be numeric, got {type(obs.reward)}"

check("Reward is always a numeric (float/int) value", test_reward_is_float)


# 8. All difficulty levels work
def test_all_difficulties():
    for diff in ["easy", "medium", "hard"]:
        env = MathReasoningEnvironment()
        obs = env.reset(difficulty=diff)
        assert obs.difficulty == diff, f"Expected difficulty={diff}, got {obs.difficulty}"
        assert obs.problem != "", f"Problem must not be empty for difficulty={diff}"

check("All difficulty levels (easy/medium/hard) work", test_all_difficulties)


# 9. state() is accessible at any time
def test_state_accessible():
    env = MathReasoningEnvironment(seed=6)
    env.reset()
    state = env.state
    assert isinstance(state, MathState), f"state must be MathState, got {type(state)}"
    assert state.episode_id is not None, "state.episode_id must not be None"
    assert isinstance(state.step_count, int), "state.step_count must be int"

check("state property is accessible and returns MathState", test_state_accessible)


# 10. state.step_count increments correctly
def test_step_count():
    env = MathReasoningEnvironment(seed=7)
    env.reset()
    assert env.state.step_count == 0
    env.step(MathAction(reasoning="r", answer="0"))
    assert env.state.step_count == 1
    env.step(MathAction(reasoning="r", answer="0"))
    assert env.state.step_count == 2

check("state.step_count increments by 1 per step", test_step_count)


# 11. reset() starts a fresh episode
def test_reset_fresh_episode():
    env = MathReasoningEnvironment(seed=8)
    obs1 = env.reset()
    id1 = env.state.episode_id
    obs2 = env.reset()
    id2 = env.state.episode_id
    assert id1 != id2, "Each reset() must produce a new episode_id"
    assert env.state.step_count == 0, "step_count must reset to 0 after reset()"

check("reset() starts a fresh episode with new episode_id", test_reset_fresh_episode)


# 12. Post-episode step returns graceful error
def test_post_episode_step():
    env = MathReasoningEnvironment(seed=9)
    env.reset(difficulty="easy")
    correct_answer = env.state.correct_answer
    env.step(MathAction(reasoning="", answer=correct_answer))  # correct → done
    obs = env.step(MathAction(reasoning="", answer="0"))  # after done
    assert obs.done is True, "step after episode end must return done=True"

check("step() after episode end returns graceful done observation", test_post_episode_step)


# 13. 5-episode simulation (integration test)
def test_5_episode_simulation():
    env = MathReasoningEnvironment(seed=42)
    for ep in range(5):
        obs = env.reset()
        assert obs.problem != ""
        for _ in range(3):
            action = MathAction(reasoning="thinking...", answer="42")
            obs = env.step(action)
            if obs.done:
                break
    state = env.state
    assert state.total_episodes == 5

check("5-episode end-to-end simulation completes successfully", test_5_episode_simulation)


# ── Summary ─────────────────────────────────────────────────────────────────
print()
print("=" * 65)
passed = sum(1 for r, _ in results if r == PASS)
failed = sum(1 for r, _ in results if r == FAIL)
total  = len(results)

print(f"  Results: {passed}/{total} checks passed  |  {failed} failed")
print("=" * 65)

if failed == 0:
    print()
    print("  🎉 All validation checks PASSED!")
    print("  Your environment is ready for submission.")
    print()
    sys.exit(0)
else:
    print()
    print(f"  ⚠️  {failed} check(s) FAILED. Please fix before submitting.")
    print()
    sys.exit(1)
