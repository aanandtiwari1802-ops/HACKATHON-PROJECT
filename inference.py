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
    # Useful for testing the Hugging Face space remotely.
    env_url = os.environ.get("ENV_URL", "http://localhost:8000")
    print(f"Connecting to OpenEnv environment at: {env_url}\n")
    
    # Use the synchronous client wrapper
    with MathReasoningEnv(base_url=env_url).sync() as env:
        # 1. Reset the environment
        result = env.reset(difficulty="easy")
        print(f"New Problem: {result.observation.problem}")
        
        # 2. Simulate model inference loop
        for step in range(3):
            print(f"\n--- Step {step+1} ---")
            
            # Dummy reasoning and answer for inference test
            dummy_action = MathAction(
                reasoning=f"I'm thinking step-by-step for attempt {step+1}...",
                answer="42"
            )
            
            # Submit the action
            result = env.step(dummy_action)
            
            print(f"Feedback: {result.observation.feedback}")
            print(f"Reward: {result.reward}")
            print(f"Done: {result.done}")
            
            if result.done:
                print("Episode finished!")
                break
                
        # 3. Print final episode state
        final_state = env.state()
        print(f"\nFinal Score: {final_state.score}")
        print(f"Total Episodes: {final_state.total_episodes}")

if __name__ == "__main__":
    main()
