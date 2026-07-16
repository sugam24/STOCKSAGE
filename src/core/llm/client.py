import os
from google import genai
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

def get_claude_response(prompt: str, system_prompt: str = "You are a helpful financial assistant.") -> str:
    """
    Sends a single prompt to Google Gemini using the official Google GenAI SDK.
    Note: We keep the function name 'get_claude_response' or make a generic helper so
    we don't have to break downstream code references as we swap models.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

    # Initialize Gemini Client (automatically configures with GEMINI_API_KEY)
    client = genai.Client(api_key=api_key)

    try:
        # Request a response using the highly efficient gemini-3.5-flash
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt,
            config={
                'system_instruction': system_prompt,
                'max_output_tokens': 1000
            }
        )

        # Print usage metrics for your Day 2 deliverable tracking
        if response.usage_metadata:
            print("\n--- Usage Metadata ---")
            print(f"Input Tokens:  {response.usage_metadata.prompt_token_count}")
            print(f"Output Tokens: {response.usage_metadata.candidates_token_count}")
            print("-----------------------\n")

        return response.text

    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {e}")
        raise e

if __name__ == "__main__":
    # Test block to verify execution
    test_prompt = "What is a stock split, and how does it affect a retail investor?"
    print(f"Sending prompt to Gemini: '{test_prompt}'...")
    try:
        reply = get_claude_response(test_prompt)
        print("Gemini's Response:")
        print(reply)
    except Exception as e:
        print(f"Failed to get response: {e}")