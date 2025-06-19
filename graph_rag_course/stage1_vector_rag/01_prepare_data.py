"""Download a tiny corpus of Wikipedia pages.
Saved under ./data/*.txt
"""
from pathlib import Path
import wikipedia

ARTICLES = [
    "Ada Lovelace",
    "Guido van Rossum",
    "Python (programming language)",
    "Alan Turing",
    "Grace Hopper",
    "Artificial intelligence",
    "Machine learning",
    "Natural language processing",
    "Chatbot",
    "OpenAI",
]

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

for title in ARTICLES:
    print(f"• {title}")
    try:
        page = wikipedia.page(title, auto_suggest=False)
        (DATA_DIR / f"{title.replace(' ', '_')}.txt").write_text(page.content, encoding="utf-8")
    except Exception as e:
        print("  × skip (", e, ")")
