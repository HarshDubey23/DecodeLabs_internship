# responses.py
# Predefined response dictionary - the "Knowledge Base" for the rule-based chatbot

responses = {
    "hello": "Hi there! How can I help you today?",
    "hi": "Hello! Nice to see you.",
    "how are you": "I'm just a program, but I'm doing great! How about you?",
    "what is your name": "I'm a Rule-Based AI Chatbot built for the DecodeLabs AI internship.",
    "what can you do": "I can chat with you using simple if-else logic and predefined rules.",
    "help": "You can say hello, ask my name, ask how I am, or type 'bye' to exit.",
    "thank you": "You're welcome!",
    "thanks": "You're welcome!",
    "bye": "Goodbye! Have a great day.",
}

FALLBACK_RESPONSE = "I'm sorry, I don't understand that yet. Type 'help' to see what I can do."
EXIT_COMMANDS = {"bye", "exit", "quit"}
