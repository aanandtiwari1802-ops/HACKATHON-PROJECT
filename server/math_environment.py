"""
Math Reasoning Environment — Server-Side Implementation.

An RL environment where an agent is given math problems
(arithmetic, algebra, word problems) and must reason step-by-step
to produce the correct numerical answer.

Reward Structure:
  +1.0  → Correct answer on first attempt
  +0.5  → Correct answer on second attempt
  +0.2  → Correct answer on third attempt
  -0.1  → Wrong answer (partial credit for valid reasoning)
  -0.5  → Episode ended with no correct answer

Difficulty Levels:
  easy   → single-step arithmetic
  medium → multi-step arithmetic / basic algebra
  hard   → multi-step algebra / word problems
"""

import math
import random
import re
import uuid
from typing import Any, Optional

try:
    from openenv.core import Environment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from dataclasses import dataclass, field
    from typing import Dict

    class Environment:
        pass

    @dataclass
    class Action:
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class Observation:
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class State:
        episode_id: Optional[str] = None
        step_count: int = 0

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MathAction, MathObservation, MathState


# ---------------------------------------------------------------------------
# Problem Bank
# ---------------------------------------------------------------------------

PROBLEM_BANK = {
    "easy": [
        {
            "problem": "What is 47 + 38?",
            "answer": "85",
            "hint": "Try adding the tens and units separately: 40+30=70, 7+8=15.",
        },
        {
            "problem": "What is 156 - 79?",
            "answer": "77",
            "hint": "156 - 80 = 76, then add 1 back since we subtracted one too many.",
        },
        {
            "problem": "What is 9 × 8?",
            "answer": "72",
            "hint": "9 × 8 = 9 × (10 - 2) = 90 - 18 = 72.",
        },
        {
            "problem": "What is 144 ÷ 12?",
            "answer": "12",
            "hint": "Think: 12 × 12 = 144.",
        },
        {
            "problem": "What is 25% of 80?",
            "answer": "20",
            "hint": "25% = 1/4. So divide 80 by 4.",
        },
        {
            "problem": "What is the square root of 81?",
            "answer": "9",
            "hint": "Which number multiplied by itself gives 81?",
        },
        {
            "problem": "What is 2³ (2 to the power of 3)?",
            "answer": "8",
            "hint": "2³ = 2 × 2 × 2.",
        },
        {
            "problem": "What is the remainder when 37 is divided by 5?",
            "answer": "2",
            "hint": "37 = 5 × 7 + remainder. 5 × 7 = 35.",
        },
    ],
    "medium": [
        {
            "problem": (
                "Solve for x: 3x + 7 = 22. What is the value of x?"
            ),
            "answer": "5",
            "hint": "Subtract 7 from both sides first, then divide by 3.",
        },
        {
            "problem": (
                "A rectangle has a length of 12 cm and a width of 8 cm. "
                "What is its area in cm²?"
            ),
            "answer": "96",
            "hint": "Area of rectangle = length × width.",
        },
        {
            "problem": (
                "A train travels 240 km in 3 hours. "
                "What is its average speed in km/h?"
            ),
            "answer": "80",
            "hint": "Speed = Distance ÷ Time.",
        },
        {
            "problem": (
                "What is the sum of the first 10 positive integers "
                "(1 + 2 + 3 + ... + 10)?"
            ),
            "answer": "55",
            "hint": "Use the formula n(n+1)/2 where n=10.",
        },
        {
            "problem": (
                "If 5 apples cost ₹35, how much will 8 apples cost (in ₹)?"
            ),
            "answer": "56",
            "hint": "Find the cost of 1 apple first, then multiply.",
        },
        {
            "problem": "Solve: 2x - 5 = 3x + 1. What is x?",
            "answer": "-6",
            "hint": "Move all x terms to one side and constants to the other.",
        },
        {
            "problem": (
                "A circle has radius 7 cm. "
                "What is its circumference? (Use π ≈ 22/7, answer in cm)"
            ),
            "answer": "44",
            "hint": "Circumference = 2πr. With π≈22/7 and r=7: 2×(22/7)×7.",
        },
        {
            "problem": (
                "What is the LCM (Least Common Multiple) of 12 and 18?"
            ),
            "answer": "36",
            "hint": "LCM(12,18) = 12×18 / GCD(12,18). GCD(12,18)=6.",
        },
    ],
    "hard": [
        {
            "problem": (
                "Two pipes A and B can fill a tank in 12 and 15 minutes "
                "respectively. Both are opened together. "
                "How many minutes will it take to fill the tank? "
                "Give your answer as a whole number."
            ),
            "answer": "7",
            "hint": (
                "Combined rate = 1/12 + 1/15 = 5/60 + 4/60 = 9/60 = 3/20 per minute. "
                "Time = 20/3 ≈ 6.67 → round to 7."
            ),
        },
        {
            "problem": (
                "A merchant buys goods for ₹800 and sells them at a 25% profit. "
                "What is the selling price in ₹?"
            ),
            "answer": "1000",
            "hint": "Selling price = Cost price × (1 + profit%). 800 × 1.25.",
        },
        {
            "problem": (
                "Solve the quadratic: x² - 5x + 6 = 0. "
                "What is the smaller root?"
            ),
            "answer": "2",
            "hint": "Factor: (x-2)(x-3)=0, so x=2 or x=3.",
        },
        {
            "problem": (
                "A boat travels 36 km upstream in 4 hours and 48 km downstream "
                "in 4 hours. What is the speed of the stream in km/h?"
            ),
            "answer": "3",
            "hint": (
                "Upstream speed = 36/4 = 9. Downstream speed = 48/4 = 12. "
                "Stream speed = (12-9)/2."
            ),
        },
        {
            "problem": (
                "In a class of 40 students, 60% are girls. "
                "How many boys are there?"
            ),
            "answer": "16",
            "hint": "Girls = 60% of 40 = 24. Boys = 40 - 24.",
        },
        {
            "problem": (
                "If log₁₀(100) = x, what is x?"
            ),
            "answer": "2",
            "hint": "log₁₀(100) = log₁₀(10²) = 2.",
        },
        {
            "problem": (
                "A sum doubles itself in 8 years at simple interest. "
                "What is the annual rate of interest (in %)?"
            ),
            "answer": "12.5",
            "hint": (
                "SI = P (so interest = principal). SI = P×R×T/100 → "
                "P = P×R×8/100 → R = 100/8 = 12.5."
            ),
        },
        {
            "problem": (
                "A 10-digit number is formed using each digit 0–9 exactly once. "
                "How many such numbers do NOT start with 0? "
                "Express your answer as an integer."
            ),
            "answer": "3265920",
            "hint": (
                "Total = 10! = 3628800. Numbers starting with 0 = 9! = 362880. "
                "Answer = 3628800 - 362880."
            ),
        },
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(answer: str) -> str:
    """Normalize an answer string for comparison."""
    answer = answer.strip().lower()
    # Remove trailing zeros after decimal (e.g. "2.50" → "2.5", "3.0" → "3")
    try:
        num = float(answer)
        if num == int(num):
            return str(int(num))
        return f"{num:.6g}"
    except ValueError:
        pass
    return answer


def _is_correct(submitted: str, ground_truth: str) -> bool:
    """Return True if submitted answer matches ground truth (numeric-aware)."""
    return _normalize(submitted) == _normalize(ground_truth)


def _reward_for_attempt(attempt_number: int, correct: bool) -> float:
    """Calculate reward based on correctness and attempt number.

    Scores must be strictly between 0 and 1 (validator requirement),
    so rewards are clamped to [0.01, 0.99].
    """
    if not correct:
        return 0.01  # was -0.1; negatives are out-of-range for score field
    rewards = {1: 0.99, 2: 0.75, 3: 0.50}
    return rewards.get(attempt_number, 0.25)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MathReasoningEnvironment(Environment):
    """
    Math Reasoning RL Environment.

    Each episode presents a math problem to an RL agent.
    The agent can attempt to solve it up to `max_attempts` times.
    Reward is shaped based on correctness and number of attempts used.
    """

    MAX_STEPS_PER_EPISODE = 3  # attempts per problem

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._state = MathState(episode_id=str(uuid.uuid4()), step_count=0)
        self._current_problem: dict = {}
        self._attempts: int = 0
        self._episode_done: bool = False
        self._total_episodes: int = 0
        self._correct_episodes: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        **kwargs: Any,
    ) -> MathObservation:
        """Start a new episode with a fresh math problem."""
        if seed is not None:
            self._rng.seed(seed)

        # Pick difficulty
        diff = difficulty or self._rng.choice(["easy", "medium", "hard"])
        problems = PROBLEM_BANK[diff]
        self._current_problem = self._rng.choice(problems)
        self._attempts = 0
        self._episode_done = False
        self._total_episodes += 1

        self._state = MathState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            correct_answer=self._current_problem["answer"],
            problem=self._current_problem["problem"],
            category="word_problem" if "?" in self._current_problem["problem"] else "algebra",
            difficulty=diff,
            total_episodes=self._total_episodes,
            correct_episodes=self._correct_episodes,
            score=(
                self._correct_episodes / self._total_episodes
                if self._total_episodes > 0
                else 0.0
            ),
        )

        return MathObservation(
            problem=self._current_problem["problem"],
            feedback="",
            correct=False,
            attempts_remaining=self.MAX_STEPS_PER_EPISODE,
            hint="",
            category=self._state.category,
            difficulty=diff,
            done=False,
            reward=0.0,
            metadata={
                "episode_id": self._state.episode_id,
                "message": "New episode started. Good luck!",
            },
        )

    def step(self, action: MathAction, **kwargs: Any) -> MathObservation:
        """
        Process one agent attempt and return an observation with reward.

        Args:
            action: MathAction with `reasoning` (chain-of-thought) and `answer`

        Returns:
            MathObservation with feedback, correctness flag, reward, and done flag
        """
        if self._episode_done:
            return MathObservation(
                problem=self._current_problem.get("problem", ""),
                feedback="Episode already finished. Please call reset().",
                correct=False,
                attempts_remaining=0,
                done=True,
                reward=0.0,
                metadata={"error": "episode_done"},
            )

        self._attempts += 1
        self._state.step_count += 1

        correct = _is_correct(action.answer, self._current_problem["answer"])
        reward = _reward_for_attempt(self._attempts, correct)

        attempts_remaining = self.MAX_STEPS_PER_EPISODE - self._attempts

        if correct:
            self._correct_episodes += 1
            self._episode_done = True
            self._state.correct_episodes = self._correct_episodes
            raw_score = self._correct_episodes / self._total_episodes
            self._state.score = min(max(raw_score, 0.01), 0.99)  # strictly (0, 1)
            feedback = (
                f"✅ Correct! The answer is {self._current_problem['answer']}. "
                f"You solved it in {self._attempts} attempt(s). "
                f"Reward: +{reward}"
            )
        elif attempts_remaining == 0:
            self._episode_done = True
            reward = -0.5
            feedback = (
                f"❌ Incorrect. The correct answer was {self._current_problem['answer']}. "
                f"You used all {self.MAX_STEPS_PER_EPISODE} attempts. Reward: {reward}"
            )
        else:
            hint = (
                self._current_problem["hint"]
                if self._attempts >= 2
                else ""
            )
            feedback = (
                f"❌ Incorrect answer '{action.answer}'. "
                f"You have {attempts_remaining} attempt(s) remaining."
                + (f" Hint: {hint}" if hint else "")
            )

        return MathObservation(
            problem=self._current_problem["problem"],
            feedback=feedback,
            correct=correct,
            attempts_remaining=attempts_remaining,
            hint=self._current_problem["hint"] if self._attempts >= 2 else "",
            category=self._state.category,
            difficulty=self._state.difficulty,
            done=self._episode_done,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "attempt_number": self._attempts,
                "reasoning_length": len(action.reasoning),
            },
        )

    @property
    def state(self) -> MathState:
        """Return current episode state."""
        return self._state

    def close(self) -> None:
        """Clean up resources, if any. Required by OpenEnv core validation."""
        pass
