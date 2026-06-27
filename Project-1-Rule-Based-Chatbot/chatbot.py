"""
Project 1 - Rule-Based AI Chatbot
DecodeLabs AI Internship

A simple chatbot that uses dictionary/if-else logic to respond to
predefined user inputs. Runs in a continuous loop until the user types
an exit command (bye / exit / quit).
"""

import sys

from responses import responses, FALLBACK_RESPONSE, EXIT_COMMANDS

# A short scripted conversation used only for automated self-testing,
# so this file can be verified to run correctly without manual typing.
TEST_CONVERSATION = ["hello", "what is your name", "help", "thank you", "bye"]


def get_response(user_input: str) -> str:
    """Clean the input and look it up in the knowledge base."""
    clean_input = user_input.lower().strip()
    return responses.get(clean_input, FALLBACK_RESPONSE)


def run_chatbot(test_mode: bool = False):
    print("=" * 50)
    print(" DecodeLabs Rule-Based AI Chatbot")
    print(" Type 'help' to see what I can do, or 'bye' to exit.")
    print("=" * 50)

    scripted_inputs = iter(TEST_CONVERSATION) if test_mode else None

    while True:
        if test_mode:
            raw_input_text = next(scripted_inputs, None)
            if raw_input_text is None:
                break
            print(f"You: {raw_input_text}")
        else:
            raw_input_text = input("You: ")

        clean_input = raw_input_text.lower().strip()

        if clean_input in EXIT_COMMANDS:
            print("Bot: Goodbye! Have a great day.")
            break

        reply = get_response(clean_input)
        print(f"Bot: {reply}")


if __name__ == "__main__":
    run_chatbot(test_mode="--test" in sys.argv)
