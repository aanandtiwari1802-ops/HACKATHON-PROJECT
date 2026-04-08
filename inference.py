"""
Inference script for the Math Reasoning Environment.

This script uses the OpenEnv client to interact with the environment,
simulating model inference for testing/validation purposes.
"""

import os
import sys

# Ensure local imports work if not installed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from math_reasoning_env.client import MathReasoningEnv
    from math_reasoning_env.models import MathAction
except ImportError:
    from client import MathReasoningEnv
    from models import MathAction

def main():
    # Allow overriding the environment URL via environment variables
    env_url = os.environ.get("ENV_URL", "http://localhost:8000")
    print(f"Connecting to OpenEnv environment at: {env_url}", file=sys.stderr, flush=True)
    
    # Use the synchronous client wrapper
    try:
        # Use a task name that matches the environment
        task_name = "math_reasoning_env"
        
        # [START] block must be first in stdout
        print(f"[START] task_id={task_name}", flush=True)
        
        with MathReasoningEnv(base_url=env_url).sync() as env:
            # 1. Reset the environment
            result = env.reset(difficulty="easy")
            print(f"New Problem: {result.observation.problem}", file=sys.stderr, flush=True)
            
            # 2. Simulate model inference loop
            step_num = 0
            total_reward = 0.0
            for i in range(3):
                step_num = i + 1
                
                # Dummy reasoning and answer for inference test
                dummy_action = MathAction(
                    reasoning=f"Step-by-step thinking for attempt {step_num}...",
                    answer="42"  # This will likely be wrong but that's fine for inference tests
                )
                
                # Submit the action
                result = env.step(dummy_action)
                step_reward = result.reward if result.reward is not None else 0.0
                total_reward += step_reward
                
                # [STEP] block in stdout
                print(f"[STEP] step={step_num} reward={step_reward}", flush=True)
                
                # Logs to stderr
                print(f"Feedback: {result.observation.feedback}", file=sys.stderr, flush=True)
                print(f"Done: {result.done}", file=sys.stderr, flush=True)
                
                if result.done:
                    break
                    
            # 3. Final state and [END] block
            final_state = env.state()
            # [END] block in stdout: use task_id and total_reward
            print(f"[END] task_id={task_name} reward={total_reward} steps={step_num}", flush=True)
            
            print(f"Final Score: {final_state.score}", file=sys.stderr, flush=True)

    except Exception as e:
        # Error logs to stderr
        if hasattr(e, 'response') and e.response is not None:
            print(f"HTTP Error: {e.response.status_code}", file=sys.stderr, flush=True)
            print(f"Detail: {e.response.text}", file=sys.stderr, flush=True)
        else:
            print(f"Error: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
