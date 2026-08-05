import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

MODEL = "llama-3.3-70b-versatile"


def get_llm_response(prompt: str, system_prompt: str = "You are a helpful financial assistant.") -> str:
    """
    Sends a single prompt to Groq using the official Groq SDK.

    Args:
        prompt:        The user's question or instruction.
        system_prompt: System-level instruction for the model persona.

    Returns:
        The model's text response.

    Raises:
        ValueError: If GROQ_API_KEY is not set.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment variables.")

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
        )

        # Print usage metrics for Day 2 deliverable tracking
        if response.usage:
            print("\n--- Usage Metadata ---")
            print(f"Input Tokens:  {response.usage.prompt_tokens}")
            print(f"Output Tokens: {response.usage.completion_tokens}")
            print("-----------------------\n")

        return response.choices[0].message.content or ""

    except Exception as e:
        print(f"An error occurred while calling the Groq API: {e}")
        raise e

if __name__ == "__main__":
    # Test block to verify execution
    test_prompt = "What is a stock split, and how does it affect a retail investor?"
    print(f"Sending prompt to Groq: '{test_prompt}'...")
    try:
        reply = get_llm_response(test_prompt)
        print("Groq's Response:")
        print(reply)
    except Exception as e:
        print(f"Failed to get response: {e}")