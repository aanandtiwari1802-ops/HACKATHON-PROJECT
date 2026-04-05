# Math Reasoning Environment 🧮

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-orange)](LICENSE)
[![Hackathon](https://img.shields.io/badge/Meta%20PyTorch-Hackathon%20'26-red)](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/)

An **OpenEnv-compatible Reinforcement Learning environment** for training LLMs to solve multi-step math problems through chain-of-thought reasoning.

---

## 🎯 Overview

The **Math Reasoning Environment** presents an RL agent with math problems spanning three difficulty levels and three topic categories. The agent must produce:
1. A **chain-of-thought reasoning** explanation
2. A **final numeric answer**

The environment grades the answer, provides feedback and optional hints, and shapes rewards based on correctness and attempt efficiency — designed for GRPO/PPO training with frameworks like TRL, torchforge, or SkyRL.

---

## 🏗️ Architecture

```
math_reasoning_env/
├── __init__.py              # Public API exports
├── models.py                # Type-safe Action / Observation / State
├── client.py                # HTTP client (EnvClient subclass)
├── sample_interface.py      # Web interface script
├── pre_validation.py        # Hackathon pre-validation script
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Package config & dependencies
├── LICENSE                  # BSD 3-Clause
└── server/
    ├── __init__.py
    ├── math_environment.py  # Core environment logic + problem bank
    ├── app.py               # FastAPI server
    ├── Dockerfile           # Container definition
    └── requirements.txt     # Server dependencies
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd math_reasoning_env
pip install fastapi uvicorn requests
# For full OpenEnv integration:
pip install openenv-core>=0.2.2
```

### 2. Run the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 3. Try it out

```python
import requests

# Reset — get a new problem
r = requests.post("http://localhost:8000/reset", params={"difficulty": "medium"})
obs = r.json()["observation"]
print(obs["problem"])
# → "A train travels 240 km in 3 hours. What is its average speed in km/h?"

# Step — submit reasoning + answer
r = requests.post("http://localhost:8000/step", json={
    "reasoning": "Speed = Distance / Time = 240 / 3 = 80",
    "answer": "80"
})
result = r.json()
print(result["observation"]["feedback"])   # ✅ Correct! ...
print(result["reward"])                    # 1.0
```

### 4. Use the Python client

```python
# Synchronous
from math_reasoning_env import MathReasoningEnv, MathAction

with MathReasoningEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(difficulty="hard")
    print(result.observation.problem)

    result = env.step(MathAction(
        reasoning="Upstream=9, Downstream=12, stream=(12-9)/2=1.5",
        answer="3"
    ))
    print(result.observation.feedback)
    print(f"Reward: {result.reward}")

# Asynchronous (recommended for RL loops)
import asyncio
from math_reasoning_env import MathReasoningEnv, MathAction

async def main():
    async with MathReasoningEnv(base_url="http://localhost:8000") as env:
        result = await env.reset()
        result = await env.step(MathAction(reasoning="...", answer="80"))
        print(result.observation.feedback)

asyncio.run(main())
```

---

## 📋 Action & Observation Spec

### MathAction (what the agent sends)

| Field | Type | Description |
|-------|------|-------------|
| `reasoning` | `str` | Chain-of-thought explanation (any length) |
| `answer` | `str` | Final numeric answer as a string |
| `metadata` | `dict` | Optional additional metadata |

### MathObservation (what the agent receives)

| Field | Type | Description |
|-------|------|-------------|
| `problem` | `str` | The math problem text |
| `feedback` | `str` | Feedback on the previous attempt |
| `correct` | `bool` | Whether the last answer was correct |
| `attempts_remaining` | `int` | How many attempts are left |
| `hint` | `str` | Hint shown after 2nd wrong attempt |
| `category` | `str` | Problem category |
| `difficulty` | `str` | `"easy"`, `"medium"`, or `"hard"` |
| `done` | `bool` | Episode ended flag |
| `reward` | `float` | Reward for this step |
| `metadata` | `dict` | Episode info |

### Reward Structure

| Event | Reward |
|-------|--------|
| Correct on **1st** attempt | `+1.0` |
| Correct on **2nd** attempt | `+0.5` |
| Correct on **3rd** attempt | `+0.2` |
| Wrong answer | `-0.1` |
| Episode fail (no correct) | `-0.5` (final step) |

---

## 🎮 Difficulty Levels

| Level | Description | Examples |
|-------|-------------|---------|
| `easy` | Single-step arithmetic | `47 + 38`, `144 ÷ 12`, `√81` |
| `medium` | Multi-step / basic algebra | `3x + 7 = 22`, speed-distance-time |
| `hard` | Multi-step algebra / word problems | Pipes & cisterns, quadratics, log |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit an action |
| `GET` | `/state` | Get current episode state |
| `GET` | `/health` | Health check |

### Reset Parameters (query string)
- `difficulty` — `"easy"` \| `"medium"` \| `"hard"` (optional, random if omitted)
- `seed` — integer seed for reproducibility (optional)

---

## 🌐 Web Interface

```bash
ENABLE_WEB_INTERFACE=true python sample_interface.py
# Open:  http://localhost:8000/web
```

The web UI provides:
- **Left pane**: Submit reasoning + answer (interactive form)
- **Right pane**: Live observation state (problem, feedback, score)
- **History log**: Full action history with rewards

---

## ✅ Pre-Validation

Run all hackathon checks locally before submitting:

```bash
python pre_validation.py
```

Expected output:
```
=================================================================
  Math Reasoning Environment — Pre-Validation Script
=================================================================

  ✅ PASS  Environment can be instantiated
  ✅ PASS  reset() returns valid MathObservation
  ✅ PASS  step() returns (obs, reward, terminated, truncated, info)
  ✅ PASS  Multiple steps (3) complete without crashing
  ✅ PASS  Episode terminates with done=True on correct answer
  ✅ PASS  Episode terminates after max attempts with negative reward
  ✅ PASS  Reward is always a numeric (float/int) value
  ✅ PASS  All difficulty levels (easy/medium/hard) work
  ✅ PASS  state property is accessible and returns MathState
  ✅ PASS  state.step_count increments by 1 per step
  ✅ PASS  reset() starts a fresh episode with new episode_id
  ✅ PASS  step() after episode end returns graceful done observation
  ✅ PASS  5-episode end-to-end simulation completes successfully

  Results: 13/13 checks passed  |  0 failed
=================================================================

  🎉 All validation checks PASSED!
```

---

## 🐳 Docker

```bash
# Build image
docker build -t math-reasoning-env:latest -f server/Dockerfile .

# Run container
docker run -p 8000:8000 math-reasoning-env:latest

# Test health
curl http://localhost:8000/health
```

---

## 📊 Use with RL Training

### Integration with TRL (GRPO)

```python
from math_reasoning_env import MathReasoningEnv, MathAction

def rollout(prompt, trainer):
    with MathReasoningEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset(difficulty="medium")
        problem = result.observation.problem

        # Generate chain-of-thought with LLM
        reasoning, answer = model.generate(problem)

        result = env.step(MathAction(reasoning=reasoning, answer=answer))
        return result.reward
```

### Integration with torchforge / SkyRL

The environment exposes standard OpenEnv WebSocket and HTTP APIs and is
compatible with any framework that supports the OpenEnv spec.

---

## 📦 Pre-Submission Checklist

- [x] Environment passes `pre_validation.py` locally (13/13 checks)
- [x] Code is open-sourced with BSD 3-Clause license
- [x] `requirements.txt` included in `server/`
- [x] `pyproject.toml` defines all dependencies
- [x] Comprehensive `README.md` provided
- [x] `sample_interface.py` demonstrates web interface
- [x] `openenv.yaml` manifest present
- [x] Dockerfile provided for containerisation
- [x] Runs within 2 vCPUs / 8 GB RAM (no ML model dependencies)
- [x] Python 3.10+ compatible

---

## 🤝 Contributing

Issues and pull requests welcome! This environment was created for the
[Meta PyTorch OpenEnv Hackathon '26](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/).

---

## 📄 License

BSD 3-Clause — see [LICENSE](LICENSE) for details.
