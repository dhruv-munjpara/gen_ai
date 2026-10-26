"""
Comparison with Traditional AI/ML - Python Simulation
Demonstrating:
1. Traditional Rule-Based / Keyword intent matching.
2. Simulated Generative LLM conversational handling with context and emotional empathy.
"""

def traditional_bot_response(user_message: str) -> str:
    message_lower = user_message.lower()
    if "track" in message_lower:
        return "Traditional Bot: [Intent: TRACK_ORDER] Please enter your 10-digit tracking ID."
    elif "cancel" in message_lower:
        return "Traditional Bot: [Intent: CANCEL_ORDER] Orders can only be cancelled within 1 hour."
    elif "return" in message_lower:
        return "Traditional Bot: [Intent: RETURN_ORDER] Check return window under My Orders."
    else:
        return "Traditional Bot: [Fallback] Sorry, I didn't understand. Please choose from: Track, Cancel, Return."

def genai_agent_response(user_message: str) -> str:
    # Simulating LLM semantic reasoning & empathetic tool-assisted generation
    response = (
        "GenAI Agent: I understand how urgent this is! I have retrieved your shipment details. "
        "Your order is currently out with our delivery partner and scheduled for delivery between "
        "4:30 PM and 6:00 PM today. I've also notified the local delivery hub to ensure prompt arrival."
    )
    return response

if __name__ == "__main__":
    test_query = "Hey, my sister's birthday is tonight and my order #FK99482 hasn't arrived yet! Can you expedite it please?"
    print("USER QUERY:")
    print(f"'{test_query}'\n")
    print("-" * 65)
    print("1. TRADITIONAL SYSTEM OUTPUT:")
    print(traditional_bot_response(test_query))
    print("-" * 65)
    print("2. GENAI SYSTEM OUTPUT:")
    print(genai_agent_response(test_query))
    print("=" * 65)
