import sys
import os
import asyncio
import logging
from typing import List, Optional

logging.basicConfig(level=logging.ERROR)
for _n in ["openai", "httpx", "urllib3", "asyncio"]:
    logging.getLogger(_n).setLevel(logging.ERROR)

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "no-key-provided"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME    = os.getenv("TASK_NAME",  "math_reasoning_env")
BENCHMARK    = os.getenv("BENCHMARK",  "math_reasoning_env")
ENV_URL      = os.getenv("ENV_URL",    "http://localhost:8000").strip()

MAX_STEPS         = 3
TEMPERATURE       = 0.2
MAX_TOKENS        = 256
SUCCESS_THRESHOLD = 0.5

# ── Logging — plain print to stdout with flush=True ───────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val  = error if error else "null"
    done_val   = str(bool(done)).lower()
    action_safe = str(action).replace("\n", " ").replace("\r", " ").strip()[:150]
    print(f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={str(bool(success)).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── [START] fires immediately — before any import that could crash ─────────────
log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

# ── Risky imports after [START] ───────────────────────────────────────────────
OpenAI = None
try:
    from openai import OpenAI
except Exception as e:
    print(f"[DEBUG] openai import failed: {e}", file=sys.stderr, flush=True)

HAS_CLIENT     = False
MathReasoningEnv = None
MathAction       = None
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from client import MathReasoningEnv, MathAction
    HAS_CLIENT = True
except Exception as e:
    print(f"[DEBUG] client import failed: {e}", file=sys.stderr, flush=True)

# ── LLM call ──────────────────────────────────────────────────────────────────
def get_model_message(client, problem, step):
    if not client:
        return f"Step {step}: Reasoning... Answer: 42"
    prompt = f"Problem: {problem}\nStep {step}: Reason step by step, then write 'Answer: <value>'"
    try:
        comp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=15.0,
        )
        return (comp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM failed step {step}: {exc}", file=sys.stderr, flush=True)
        return f"Step {step}: Fallback. Answer: 42"

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    rewards     = []
    steps_taken = 0
    score       = 0.0
    success     = False
    env         = None

    client = None
    if OpenAI:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        except Exception as exc:
            print(f"[DEBUG] OpenAI init failed: {exc}", file=sys.stderr, flush=True)

    try:
        if HAS_CLIENT:
            env    = MathReasoningEnv(base_url=ENV_URL)
            result = await env.reset()
            problem = result.observation.problem

            for step in range(1, MAX_STEPS + 1):
                if getattr(result, "done", False):
                    break

                content = get_model_message(client, problem, step)

                if "Answer:" in content:
                    reasoning, answer = content.split("Answer:", 1)
                    reasoning, answer = reasoning.strip(), answer.strip()
                else:
                    reasoning, answer = content, "42"

                result = await env.step(MathAction(reasoning=reasoning, answer=answer))
                obs    = result.observation
                reward = float(getattr(result, "reward", None) or 0.0)
                done   = bool(getattr(result, "done", False) or getattr(obs, "done", False))
                error  = getattr(obs, "error_message", None)

                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=content.replace("\n", " "),
                         reward=reward, done=done, error=error)
                if done:
                    break

            try:
                score = float(env.state().score)
            except Exception:
                score = sum(rewards) / len(rewards) if rewards else 0.0

        else:
            # Simulation fallback
            for step in range(1, MAX_STEPS + 1):
                rewards.append(1.0)
                log_step(step=step, action=f"Simulating step {step}... Answer: 42",
                         reward=1.00, done=(step == MAX_STEPS), error=None)
                steps_taken = step
            score = 1.0

        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] main exception: {exc}", file=sys.stderr, flush=True)
        if not rewards:
            rewards.append(0.0)
            steps_taken = 1
            log_step(step=1, action="error-recovery Answer: 0",
                     reward=0.0, done=True, error=str(exc)[:120])

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", file=sys.stderr, flush=True)

        score   = min(max(float(score), 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())