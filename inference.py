"""
Inference Script for Math Reasoning Environment
================================================
STDOUT FORMAT:
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

# ── Print [START] immediately — before ANY other imports or logic ─────────────
import os, sys
print(
    f"[START]"
    f" task={os.getenv('TASK_NAME','math_reasoning')}"
    f" env={os.getenv('BENCHMARK','math_reasoning_env')}"
    f" model={os.getenv('MODEL_NAME','Qwen/Qwen2.5-72B-Instruct')}",
    flush=True
)

# ── Now safe to do everything else ───────────────────────────────────────────
import time
import subprocess
import textwrap
import urllib.request
import urllib.error
import json
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
    print("[DEBUG] openai not installed", file=sys.stderr, flush=True)

# ── Structured stdout helpers ─────────────────────────────────────────────────
def log_step(step, action, reward, done, error):
    a = str(action).replace("\n", " ").replace("\r", " ").strip()[:200]
    e = str(error).strip()[:120] if error else "null"
    d = "true" if done else "false"
    print(f"[STEP] step={step} action={a} reward={reward:.2f} done={d} error={e}", flush=True)

def log_end(success, steps, score, rewards):
    rs = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rs}", flush=True)

# ── HTTP helpers (stdlib only — no requests needed) ───────────────────────────
def http_post(url, payload):
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

def http_get(url):
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read())

# ── Docker helpers ────────────────────────────────────────────────────────────
def start_docker(image, port):
    try:
        r = subprocess.run(
            ["docker", "run", "-d", "--rm", "-p", f"{port}:{port}", image],
            capture_output=True, text=True, timeout=60
        )
        if r.returncode == 0:
            cid = r.stdout.strip()
            print(f"[DEBUG] container started: {cid[:12]}", file=sys.stderr, flush=True)
            return cid
        print(f"[DEBUG] docker run failed: {r.stderr.strip()}", file=sys.stderr, flush=True)
        return None
    except Exception as exc:
        print(f"[DEBUG] docker start error: {exc}", file=sys.stderr, flush=True)
        return None

def wait_for_server(url, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False

def stop_docker(cid):
    try:
        subprocess.run(["docker", "stop", cid], capture_output=True, timeout=30)
    except Exception:
        pass

# ── LLM call ─────────────────────────────────────────────────────────────────
def get_model_message(client, problem, step, last_feedback, history):
    if not _OPENAI_OK or client is None:
        return "Reasoning: fallback.\nAnswer: 42"
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
        )
        text = (completion.choices[0].message.content or "").strip()
        return text or "Answer: 42"
    except Exception as exc:
        print(f"[DEBUG] LLM failed: {exc}", file=sys.stderr, flush=True)
        return "Answer: 42"

# ── Main (synchronous — no asyncio needed) ────────────────────────────────────
def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if _OPENAI_OK else None

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    container_id = None

    try:
        # ── Start server ──────────────────────────────────────────────────────
        # First try connecting directly (in case server is already running)
        server_ready = wait_for_server(ENV_URL, timeout=5)

        if not server_ready:
            # Start via Docker
            container_id = start_docker(IMAGE_NAME, ENV_PORT)
            server_ready = wait_for_server(ENV_URL, timeout=60)

        if not server_ready:
            raise RuntimeError(f"Server at {ENV_URL} did not start")

        # ── Episode ───────────────────────────────────────────────────────────
        reset_resp    = http_post(f"{ENV_URL}/reset", {})
        obs_data      = reset_resp.get("observation", reset_resp)
        problem       = obs_data.get("problem", "What is 2 + 2?")
        last_feedback = ""
        history: List[str] = []

        for step in range(1, MAX_STEPS + 1):
            if obs_data.get("done", False):
                break

            content = get_model_message(client, problem, step, last_feedback, history)

            if "Answer:" in content:
                reasoning, answer = content.split("Answer:", 1)
                reasoning, answer = reasoning.strip(), answer.strip()
            else:
                reasoning, answer = content, "42"

            step_resp = http_post(f"{ENV_URL}/step", {
                "reasoning": reasoning,
                "answer": answer
            })
            obs_data      = step_resp.get("observation", step_resp)
            reward        = float(step_resp.get("reward", obs_data.get("reward", 0.0)) or 0.0)
            done          = bool(step_resp.get("done", obs_data.get("done", False)))
            error         = None

            if not obs_data.get("correct", True):
                fb = obs_data.get("feedback", "")
                error = fb[:80] if fb else None
            last_feedback = obs_data.get("feedback", "")

            rewards.append(reward)
            steps_taken = step
            log_step(step, content.replace("\n", " "), reward, done, error)
            history.append(f"Step {step}: answer={answer!r} reward={reward:+.2f}")

            if done:
                break

        try:
            state_data = http_get(f"{ENV_URL}/state")
            score      = float(state_data.get("score", 0.0))
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
    main()