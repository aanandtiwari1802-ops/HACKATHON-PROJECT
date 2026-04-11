"""
Inference Script for Math Reasoning Environment
================================================
STDOUT FORMAT (required):
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import asyncio
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")  or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME    = os.getenv("TASK_NAME",  "math_reasoning_env")
BENCHMARK    = os.getenv("BENCHMARK",  "math_reasoning_env")
ENV_URL      = os.getenv("ENV_URL",    "http://localhost:8000").strip()

MAX_STEPS         = 3
TEMPERATURE       = 0.2
MAX_TOKENS        = 512
SUCCESS_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
    You are a math reasoning assistant.
    For each problem, reason step by step and then provide your final answer.
    Always end your response with exactly: Answer: <value>
    where <value> is just the number or expression, nothing else.
""").strip()

# ── Structured stdout helpers ─────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_safe = str(action).replace("\n", " ").replace("\r", " ").strip()[:200]
    error_val   = str(error).strip()[:120] if error else "null"
    done_val    = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── Prompt builder ────────────────────────────────────────────────────────────
def build_user_prompt(problem: str, step: int, last_feedback: str, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(f"""
        Problem: {problem}
        Step: {step}
        Last feedback: {last_feedback or 'None'}
        Previous attempts:
        {history_block}
        Reason step by step, then write 'Answer: <value>'
    """).strip()

# ── LLM call ──────────────────────────────────────────────────────────────────
def get_model_message(client: OpenAI, problem: str, step: int,
                      last_feedback: str, history: List[str]) -> str:
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

# ── Main ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    history:     List[str]   = []
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    # Import env client (done here so [START] is always printed first)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from client import MathReasoningEnv, MathAction
    except Exception as exc:
        print(f"[DEBUG] client import failed: {exc}", file=sys.stderr, flush=True)
        # Emit a minimal fallback so validator gets [END]
        log_step(step=1, action="import-error", reward=0.0, done=True, error=str(exc)[:120])
        log_end(success=False, steps=1, score=0.0, rewards=[0.0])
        return

    try:
        with MathReasoningEnv(base_url=ENV_URL).sync() as env:
            result        = env.reset()
            problem       = result.observation.problem
            last_feedback = ""

            for step in range(1, MAX_STEPS + 1):
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
                done   = bool(getattr(result, "done", False) or getattr(obs, "done", False))
                error  = None

                if hasattr(obs, "feedback") and not getattr(obs, "correct", True) and obs.feedback:
                    error = obs.feedback.encode("ascii", "ignore").decode().strip()[:80]
                    last_feedback = obs.feedback
                else:
                    last_feedback = ""

                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=content.replace("\n", " "),
                         reward=reward, done=done, error=error)
                history.append(f"Step {step}: answer={answer!r} reward={reward:+.2f}")

                if done:
                    break

            try:
                score = float(env.state().score)
            except Exception:
                score = sum(rewards) / len(rewards) if rewards else 0.0

        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] main exception: {exc}", file=sys.stderr, flush=True)
        if not rewards:
            rewards.append(0.0)
            steps_taken = 1
            log_step(step=1, action="error-recovery", reward=0.0, done=True, error=str(exc)[:120])

    finally:
        score   = min(max(float(score), 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
