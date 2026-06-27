"""
Project 3 - AI Recommendation Logic (Tech Stack Recommender) - v2
DecodeLabs AI Internship

Content-based recommendation engine using TF-IDF + Cosine Similarity.

Improvements over v1:
  1. Skill alias normalization  (ML → Machine Learning, JS → JavaScript, etc.)
  2. Shows MATCHED skills       — which of your skills triggered each result
  3. Shows MISSING skills       — what you'd need to score higher
  4. Visual progress bar        — match shown as 0-100% bar, not raw decimal
  5. Interactive mode           — prompts for skills if none supplied via CLI
  6. --export flag              — saves Top-N results to recommendations.csv
  7. Richer items.csv           — 15 roles with broader skill tag sets
"""

import argparse
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_TOP_N = 3

# ── Alias map: shorthand → canonical skill name ────────────────────────
SKILL_ALIASES = {
    "ml":   "Machine Learning",
    "ai":   "Artificial Intelligence",
    "dl":   "Deep Learning",
    "nlp":  "Natural Language Processing",
    "cv":   "Computer Vision",
    "js":   "JavaScript",
    "ts":   "TypeScript",
    "k8s":  "Kubernetes",
    "gcp":  "Google Cloud Platform",
    "db":   "Database",
    "oop":  "Object Oriented Programming",
    "ui":   "UI Design",
    "ux":   "UX Design",
}


# ── Helpers ────────────────────────────────────────────────────────────
def normalize_skills(skills: list) -> list:
    """Expand abbreviations to their full canonical names."""
    normalized = []
    for s in skills:
        s = s.strip()
        normalized.append(SKILL_ALIASES.get(s.lower(), s))
    return normalized


def load_items(path: str = "items.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Items file not found: {path}")
    return pd.read_csv(path)


def get_matched_skills(user_skills: list, role_skills: str) -> list:
    """Which of the user's skills appear in this role's skill set?"""
    role_lower = role_skills.lower()
    return [s for s in user_skills if s.lower() in role_lower]


def get_missing_skills(user_skills: list, role_skills: str, top_n: int = 3) -> list:
    """Top skills in this role that the user does NOT currently have."""
    role_skill_list = [s.strip() for s in role_skills.split()]
    user_lower = [s.lower() for s in user_skills]
    missing = [s for s in role_skill_list if s.lower() not in user_lower]
    return missing[:top_n]


def make_bar(pct: float, width: int = 20) -> str:
    filled = int(round(pct / 100 * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


# ── Core recommendation logic ──────────────────────────────────────────
def recommend(
    user_skills: list,
    items_df: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    item_docs = items_df["skills"].tolist()
    user_doc  = " ".join(user_skills)

    # Step 1: Ingestion — vectorize items + user profile into ONE vocabulary space
    vectorizer   = TfidfVectorizer()
    item_vectors = vectorizer.fit_transform(item_docs)
    user_vector  = vectorizer.transform([user_doc])

    # Step 2: Scoring — cosine similarity between user vector and every role vector
    scores = cosine_similarity(user_vector, item_vectors).flatten()

    # Step 3: Sorting — rank by score descending
    result = items_df.copy()
    result["match_score"] = scores
    result["match_pct"]   = (scores * 100).round(1)
    result = result.sort_values(by="match_score", ascending=False)

    # Step 4: Filtering — Top-N results only
    return result.head(top_n).reset_index(drop=True)


# ── Pretty printer ─────────────────────────────────────────────────────
def print_results(results: pd.DataFrame, user_skills: list) -> None:
    for i, row in results.iterrows():
        matched = get_matched_skills(user_skills, row["skills"])
        missing = get_missing_skills(user_skills, row["skills"])
        bar     = make_bar(row["match_pct"])

        print(f"  #{i + 1}  {row['role']}")
        print(f"       Match : {bar}  {row['match_pct']}%")

        if matched:
            print(f"       Matched skills  : {', '.join(matched)}")
        if missing:
            print(f"       Skills to add   : {', '.join(missing)}")
        print()


# ── Entry point ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Project 3 - Tech Stack Recommender v2"
    )
    parser.add_argument(
        "--skills",
        default=None,
        help="Comma-separated skills, e.g. 'Python,Machine Learning,SQL'",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of recommendations (default: {DEFAULT_TOP_N})",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Save results to recommendations.csv",
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  DecodeLabs Tech Stack Recommender  v2")
    print("=" * 55)

    # Interactive mode if --skills not supplied
    if args.skills is None:
        print("  Enter your skills (comma-separated)")
        print("  e.g. Python, Machine Learning, SQL, AWS")
        raw = input("  Your skills: ")
    else:
        raw = args.skills

    # Clean + normalize
    raw_skills  = [s.strip() for s in raw.split(",") if s.strip()]
    user_skills = normalize_skills(raw_skills)

    items_df = load_items()
    results  = recommend(user_skills, items_df, top_n=args.top)

    print(f"\n  Your skills : {', '.join(user_skills)}")
    print(f"  Top {args.top} career matches:\n")
    print("=" * 55)
    print()

    print_results(results, user_skills)

    if args.export:
        out = "recommendations.csv"
        results.to_csv(out, index=False)
        print(f"  Results exported to {out}")

    print("=" * 55)


if __name__ == "__main__":
    main()