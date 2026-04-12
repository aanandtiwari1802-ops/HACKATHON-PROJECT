"""
Inference Script for Math Reasoning Environment
================================================

STDOUT FORMAT (required by validator):

  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Three complete [START]→[END] blocks are emitted — one per task:
  • easy_arithmetic
  • medium_algebra
  • hard_reasoning
"""

import os
import sys
import textwrap
from typing import List, Optional, Tuple

from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY       = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK     = os.getenv("BENCHMARK",    "math_reasoning_env")
ENV_URL       = os.getenv("ENV_URL",      "http://localhost:8000").strip()

MAX_STEPS_PER_TASK = 3   # steps allowed per individual task episode
TEMPERATURE        = 0.2
MAX_TOKENS         = 512
SUCCESS_THRESHOLD  = 0.5

# Map task_id (matches openenv.yaml task names) → env difficulty
TASK_DIFFICULTY_MAP: dict[str, str] = {
    "easy_arithmetic": "easy",
    "medium_algebra":  "medium",
    "hard_reasoning":  "hard",
}

SYSTEM_PROMPT = textwrap.dedent("""
    You are a math reasoning assistant.
    For each problem, reason step by step and then provide your final answer.
    Always end your response with exactly: Answer: <value>
    where <value> is just the number or expression, nothing else.
""").strip()

# ── Structured stdout helpers ─────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    action_safe = str(action).replace("\n", " ").replace("\r", " ").strip()[:200]
    error_val   = str(error).strip()[:120] if error else "null"
    done_val    = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action_safe} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── Prompt builder ────────────────────────────────────────────────────────────

def build_user_prompt(
    problem: str,
    step: int,
    last_feedback: str,
    history: List[str],
) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(f"""
        Problem: {problem}
        Step: {step}
        Last feedback: {last_feedback or 'None'}
        Previous attempts:
        {history_block}
        Reason step by step, then write 'Answer: <value>'
    """).strip()

# ── LLM call ─────────────────────────────────────────────────────────────────

def get_model_message(
    client: OpenAI,
    problem: str,
    step: int,
    last_feedback: str,
    history: List[str],
) -> str:
    user_prompt = build_user_prompt(problem, step, last_feedback, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "Answer: 42"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        return "Answer: 42"

# ── Single-task episode ───────────────────────────────────────────────────────

def run_episode(
    task_id: str,
    client: OpenAI,
) -> Tuple[bool, int, float, List[float]]:
    """
    Run one complete episode for *task_id*.
    Returns (success, total_steps, avg_score, all_rewards).
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from client import MathReasoningEnv, MathAction  # type: ignore[import]

    difficulty = TASK_DIFFICULTY_MAP.get(task_id, "easy")

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    history:     List[str]   = []

    with MathReasoningEnv(base_url=ENV_URL).sync() as env:
        # Reset the environment, selecting difficulty that matches this task
        result       = env.reset(difficulty=difficulty)
        problem      = result.observation.problem
        last_feedback = ""

        for step in range(1, MAX_STEPS_PER_TASK + 1):
            if getattr(result, "done", False):
                break

            content = get_model_message(client, problem, step, last_feedback, history)

            if "Answer:" in content:
                reasoning, answer = content.split("Answer:", 1)
                reasoning, answer = reasoning.strip(), answer.strip()
            else:
                reasoning, answer = content, "42"

            result = env.step(MathAction(reasoning=reasoning, answer=answer))
            obs    = result.observation
            reward = float(getattr(result, "reward", None) or 0.0)
            done   = bool(
                getattr(result, "done", False)
                or getattr(obs, "done", False)
            )

            error = None
            if (
                hasattr(obs, "feedback")
                and not getattr(obs, "correct", True)
                and obs.feedback
            ):
                error = obs.feedback.encode("ascii", "ignore").decode().strip()[:80]

            last_feedback = getattr(obs, "feedback", "") or ""

            rewards.append(reward)
            steps_taken = step
            log_step(
                step=step,
                action=content.replace("\n", " "),
                reward=reward,
                done=done,
                error=error,
            )
            history.append(f"Step {step}: answer={answer!r} reward={reward:+.2f}")

            if done:
                break

        # Try to get final score from env state
        try:
            score = float(env.state().score)
        except Exception:
            score = sum(rewards) / len(rewards) if rewards else 0.0

        success = score >= SUCCESS_THRESHOLD

    if not rewards:
        rewards = [0.0]
        steps_taken = steps_taken or 1

    score   = min(max(float(score), 0.0), 1.0)
    success = score >= SUCCESS_THRESHOLD
    return success, steps_taken, score, rewards

# ── Main: 3 task episodes ─────────────────────────────────────────────────────

def main() -> None:
    """
    Run exactly 3 episodes — one per task — each producing its own
    [START] … [STEP] … [END] block as required by the validator.
    """
    # Import client here so [START] is always printed before any crash
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from client import MathReasoningEnv, MathAction  # noqa: F401 — validate importability
    except Exception as exc:
        # Emit minimal 3×fallback blocks so the validator can count them
        for task_id in ["easy_arithmetic", "medium_algebra", "hard_reasoning"]:
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            log_step(step=1, action="import-error", reward=0.0, done=True,
                     error=str(exc)[:120])
            log_end(success=False, steps=1, score=0.0, rewards=[0.0])
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id in ["easy_arithmetic", "medium_algebra", "hard_reasoning"]:
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        success:     bool        = False
        total_steps: int         = 0
        avg_score:   float       = 0.0
        all_rewards: List[float] = []

        try:
            success, total_steps, avg_score, all_rewards = run_episode(
                task_id=task_id,
                client=client,
            )
        except Exception as exc:
            print(f"[DEBUG] Episode '{task_id}' failed: {exc}", flush=True)
            # Emit a placeholder step so the block is syntactically complete
            log_step(step=1, action="episode-error", reward=0.0, done=True,
                     error=str(exc)[:120])
            all_rewards = all_rewards or [0.0]
            total_steps = total_steps or 1
        finally:
            log_end(
                success=success,
                steps=total_steps,
                score=avg_score,
                rewards=all_rewards,
            )


if __name__ == "__main__":
    main()