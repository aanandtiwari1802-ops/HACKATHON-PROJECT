"""
Inference Script for Math Reasoning Environment
================================================
STDOUT FORMAT:
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import sys
import asyncio
import textwrap
import time
import subprocess
import urllib.request
from typing import List, Optional

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY       = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy-key"
API_BASE_URL  = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME    = os.getenv("MODEL_NAME")  or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME     = os.getenv("TASK_NAME",  "math_reasoning")
BENCHMARK     = os.getenv("BENCHMARK",  "math_reasoning_env")
IMAGE_NAME    = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME", "math-reasoning-env:latest")
ENV_PORT      = int(os.getenv("ENV_PORT", "8000"))
ENV_URL       = f"http://localhost:{ENV_PORT}"
MAX_STEPS         = 3
TEMPERATURE       = 0.2
MAX_TOKENS        = 512
SUCCESS_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
    You are a math reasoning assistant.
    Reason step by step, then end with: Answer: <value>
    <value> must be only the number or expression, nothing else.
""").strip()

# ── Guard openai import ───────────────────────────────────────────────────────
try:
    from openai import OpenAI
    _OPENAI_OK = True
except ImportError:
    _OPENAI_OK = False
    print("[DEBUG] openai not installed — using fallback", file=sys.stderr, flush=True)

# ── Structured stdout helpers ─────────────────────────────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    a = str(action).replace("\n", " ").replace("\r", " ").strip()[:200]
    e = str(error).strip()[:120] if error else "null"
    d = "true" if done else "false"
    print(f"[STEP] step={step} action={a} reward={reward:.2f} done={d} error={e}", flush=True)

def log_end(success, steps, score, rewards):
    rs = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rs}", flush=True)

# ── Docker helpers ────────────────────────────────────────────────────────────
def start_docker(image: str, port: int) -> Optional[str]:
    """Start the env server in a Docker container. Returns container ID or None."""
    try:
        result = subprocess.run(
            ["docker", "run", "-d", "--rm", "-p", f"{port}:{port}", image],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            container_id = result.stdout.strip()
            print(f"[DEBUG] Docker container started: {container_id[:12]}", file=sys.stderr, flush=True)
            return container_id
        else:
            print(f"[DEBUG] Docker run failed: {result.stderr.strip()}", file=sys.stderr, flush=True)
            return None
    except Exception as exc:
        print(f"[DEBUG] Docker start error: {exc}", file=sys.stderr, flush=True)
        return None

def wait_for_server(url: str, timeout: int = 60) -> bool:
    """Poll /health until the server is ready."""
    health_url = f"{url}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as r:
                if r.status == 200:
                    print(f"[DEBUG] Server ready at {url}", file=sys.stderr, flush=True)
                    return True
        except Exception:
            pass
        time.sleep(2)
    print(f"[DEBUG] Server did not become ready in {timeout}s", file=sys.stderr, flush=True)
    return False

def stop_docker(container_id: str) -> None:
    """Stop the Docker container."""
    try:
        subprocess.run(["docker", "stop", container_id],
                       capture_output=True, timeout=30)
        print(f"[DEBUG] Docker container stopped: {container_id[:12]}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"[DEBUG] Docker stop error: {exc}", file=sys.stderr, flush=True)

# ── LLM call ─────────────────────────────────────────────────────────────────
def get_model_message(client, problem, step, last_feedback, history):
    if not _OPENAI_OK or client is None:
        return "Reasoning: fallback mode.\nAnswer: 42"

    history_block = "\n".join(history[-4:]) if history else "None"
    user_prompt = textwrap.dedent(f"""
        Problem: {problem}
        Step: {step}
        Last feedback: {last_feedback or 'None'}
        Previous attempts:
        {history_block}
        Reason step by step, then write 'Answer: <value>'
    """).strip()

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
        return text or "Answer: 42"
    except Exception as exc:
        print(f"[DEBUG] LLM failed: {exc}", file=sys.stderr, flush=True)
        return "Answer: 42"

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    # [START] MUST be the very first line of output
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if _OPENAI_OK else None

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    container_id = None

    try:
        # ── Start Docker container ────────────────────────────────────────────
        container_id = start_docker(IMAGE_NAME, ENV_PORT)
        if container_id:
            server_ready = wait_for_server(ENV_URL, timeout=60)
        else:
            server_ready = False

        if not server_ready:
            raise RuntimeError(f"Server at {ENV_URL} did not start (image={IMAGE_NAME})")

        # ── Run episode ───────────────────────────────────────────────────────
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from client import MathReasoningEnv, MathAction

        with MathReasoningEnv(base_url=ENV_URL).sync() as env:
            result        = env.reset()
            problem       = result.observation.problem
            last_feedback = ""
            history: List[str] = []

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

                fb = getattr(obs, "feedback", "")
                if fb and not getattr(obs, "correct", True):
                    error = fb.encode("ascii", "ignore").decode().strip()[:80]
                last_feedback = fb

                rewards.append(reward)
                steps_taken = step
                log_step(step, content.replace("\n", " "), reward, done, error)
                history.append(f"Step {step}: answer={answer!r} reward={reward:+.2f}")

                if done:
                    break

            try:
                score = float(env.state().score)
            except Exception:
                score = sum(rewards) / len(rewards) if rewards else 0.0

    except Exception as exc:
        print(f"[DEBUG] episode failed: {exc}", file=sys.stderr, flush=True)
        if not rewards:
            rewards     = [0.0]
            steps_taken = 1
            log_step(1, "error-recovery", 0.0, True, str(exc)[:120])

    finally:
        if container_id:
            stop_docker(container_id)
        score   = min(max(float(score), 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD
        log_end(success, steps_taken, score, rewards)

if __name__ == "__main__":
    asyncio.run(main())