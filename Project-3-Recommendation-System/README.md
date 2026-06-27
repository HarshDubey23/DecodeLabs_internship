# Project 3 — AI Recommendation Logic (Tech Stack Recommender)

## Overview
A content-based recommendation engine that maps a user's skills to the
most relevant tech career paths (e.g., Data Scientist, DevOps Engineer,
Cloud Architect) using TF-IDF feature extraction and Cosine Similarity —
the same family of techniques used by Netflix, Amazon, and Spotify.

## Objective
Create a recommendation system that matches user preferences to items
using similarity logic rather than random suggestions.

## Features
- Vectorizes user skills + job-role skill tags into the same TF-IDF vocabulary space
- Computes Cosine Similarity between the user profile and every job role
- Ranks and filters results into a clean Top-3 (Top-N) list
- Runs from the command line with your own skill list

## Architecture (IPO Model)

INPUT                  PROCESS                       OUTPUT
User skills      -->   TF-IDF + Cosine Similarity --> Top-3 Ranked
items.csv             (Ingest->Score->Sort->Filter)  Career Paths

## Tech Stack
Python, pandas, scikit-learn

## Installation
```bash
pip install -r requirements.txt
```

## Run
```bash
# Use the example from the brief
python recommend.py --skills "Python,Cloud Computing,Automation"

# Or with your own skills
python recommend.py --skills "JavaScript,React,HTML"
```

## Screenshots
> Replace this with a real screenshot of your terminal output (top-3
> recommendations).

![Recommendation demo](screenshots/recommend_demo.png)

## Folder Structure

Project-3-Recommendation-System/
│── items.csv             # Job roles + associated skill tags
│── recommend.py          # Recommendation logic
│── requirements.txt
│── screenshots/
│── README.md

## Future Improvements
- Add a feedback loop so users can mark recommendations as relevant
- Combine with collaborative filtering for a hybrid system
- Add more job roles and richer skill tag sets
- Build a simple web UI (Streamlit/Flask) around the same logic
