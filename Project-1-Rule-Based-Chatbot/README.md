# Project 1 — Rule-Based AI Chatbot 🤖

## Overview
A rule-based chatbot that responds to predefined user inputs using simple
dictionary/if-else logic instead of machine learning. It demonstrates the
foundational building blocks of AI systems: control flow, decision-making,
and continuous interaction loops.

## Objective
Build a chatbot that simulates basic human conversation through pure
programmatic decision-making — no training, no model, just clean logic.

## Features
- Continuous input loop (`while True`) that keeps the conversation alive
- Input sanitization (case-folding + whitespace stripping)
- Knowledge base of 8+ predefined intents stored in a dictionary
- Fallback response for unrecognized input
- Clean exit on `bye` / `exit` / `quit`

## Architecture

User Input
|
v
Sanitize (lower + strip)
|
v
Dictionary Lookup (.get with fallback)
|
v
Bot Response
|
v
Loop until exit command

## Tech Stack
Python (standard library only — no external dependencies)

## Installation
No installation needed beyond Python 3.8+.

## Run
```bash
python chatbot.py
```
Try saying: `hello`, `how are you`, `help`, `thank you`, `bye`

A non-interactive self-test mode is also available (used to verify the
logic without manual typing):
```bash
python chatbot.py --test
```

## Screenshots
> Replace this with a real screenshot of your own terminal conversation.

![Chat demo](screenshots/chat_demo.png)

## Folder Structure

Project-1-Rule-Based-Chatbot/
│── chatbot.py          # Main chatbot logic
│── responses.py        # Predefined responses (knowledge base)
│── screenshots/        # Demo screenshots
│── README.md

## Future Improvements
- Expand the knowledge base with more intents
- Add fuzzy matching for typos
- Give the bot a distinct personality
- Hybrid mode: fall back to an LLM when no rule matches
