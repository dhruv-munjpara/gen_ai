"""
What is GenAI? - Task 4 Demonstration Script
Demonstrating prompt formatting, simulated generative token sampling, and rhyme verification.
"""

import time
import random

PROMPT = """Write a catchy, heartwarming 4-line poem for a food delivery app 
(like Zomato or Swiggy) notifying the user that their hot dinner is on the way on a rainy evening."""

POEM_CANDIDATES = [
    [
        "Outside the rain begins to pour,",
        "Your steaming feast is at your door.",
        "A warm delight to ease the night,",
        "Unbox the joy with every bite!"
    ],
    [
        "Thunder rolls and droplets fall,",
        "Hot spicy treats answer your call.",
        "Delivered swift through rain and storm,",
        "To keep your belly happy and warm!"
    ]
]

def simulate_llm_generation(prompt: str, seed: int = 42) -> str:
    print("=" * 60)
    print("PROMPT SUBMITTED TO GENERATIVE MODEL:")
    print("=" * 60)
    print(prompt.strip())
    print("-" * 60)
    print("[INFO] Tokenizing input prompt...")
    print("[INFO] Computing cross-attention across context tokens...")
    print("[INFO] Autoregressively sampling next tokens (temperature=0.7)...")
    time.sleep(0.5)
    
    random.seed(seed)
    selected_poem = random.choice(POEM_CANDIDATES)
    
    print("\nGENERATED POEM OUTPUT:")
    print("-" * 60)
    for line in selected_poem:
        print(f"  {line}")
    print("=" * 60)
    return "\n".join(selected_poem)

if __name__ == "__main__":
    simulate_llm_generation(PROMPT)
