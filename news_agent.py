#!/usr/bin/env python3
"""
News Agent CLI
Usage:
    python news_agent.py                     # interactive
    python news_agent.py -q "AI news today"  # single query
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

from agent import build_agent, run_agent

MODEL = "llama-3.3-70b-versatile"


def print_results(articles: list[dict], query: str):
    print(f"\n{'='*60}")
    print(f"Results for: {query}")
    print(f"{'='*60}\n")
    for i, a in enumerate(articles, 1):
        if "error" in a:
            print(f"  ERROR: {a['error']}")
            continue
        print(f"{i}. [{a.get('source', '?')}]  {a.get('published', '')}")
        print(f"   Headline : {a.get('headline', 'N/A')}")
        print(f"   Summary  : {a.get('summary', 'N/A')}")
        print(f"   URL      : {a.get('url', 'N/A')}")
        print()


def main():
    groq_key   = os.getenv("GROQ_API_KEY", "")
    tavily_key = os.getenv("TAVILY_API_KEY", "")

    if not groq_key:
        sys.exit("Set GROQ_API_KEY in your .env file.")
    if not tavily_key:
        sys.exit("Set TAVILY_API_KEY in your .env file.")

    llm_with_tools, search_tool = build_agent(groq_key, tavily_key)

    if "-q" in sys.argv:
        idx   = sys.argv.index("-q")
        query = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ""
        if not query:
            sys.exit("Provide a query after -q")
        articles = run_agent(llm_with_tools, search_tool, query)
        print_results(articles, query)
        return

    print(f"News Agent  (Groq/{MODEL} + Tavily)  |  type 'quit' to exit\n")
    while True:
        try:
            query = input("Search › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        articles = run_agent(llm_with_tools, search_tool, query)
        print_results(articles, query)


if __name__ == "__main__":
    main()
