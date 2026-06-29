#!/usr/bin/env python3
"""
News Agent — Real agentic loop
- LLM (Groq/LLaMA) decides when and how to call the Tavily web search tool
- After each tool call the LLM evaluates whether the results are good enough
- If not, it refines the query and searches again (up to MAX_ITERATIONS)
- Stops as soon as it's satisfied with the retrieved data

Usage:
    python news_agent.py                     # interactive
    python news_agent.py -q "AI news today"  # single query
"""

import os
import sys
import json
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# ── Config ────────────────────────────────────────────────────
MODEL          = "llama-3.3-70b-versatile"
MAX_ITERATIONS = 5   # max search attempts before giving up
MAX_ARTICLES   = 5

SYSTEM_PROMPT = f"""You are a news retrieval agent. Today is {datetime.now().strftime("%d %B %Y")}.

Your job:
1. Use the `tavily_search` tool to search for real, recent news on the user's topic.
2. Evaluate the results — are they relevant, recent, and from credible sources?
3. If the results are poor or off-topic, refine your search query and search again.
4. Stop searching once you have {MAX_ARTICLES} or fewer good articles.

When you are satisfied with the results, respond ONLY with a valid JSON array — no markdown, no extra text:
[
  {{
    "headline": "Full accurate headline",
    "published": "Date e.g. 29 June 2026",
    "source":    "Publisher name",
    "summary":   "2-3 sentence factual summary.",
    "url":       "https://article-url.com/"
  }}
]

Rules:
- Only include articles genuinely relevant to the topic.
- Never fabricate headlines, dates, URLs, or facts — use only what the tool returned.
- If after all searches nothing relevant was found: [{{"error": "No relevant news found."}}]
"""


def build_agent(groq_key: str, tavily_key: str):
    """Return the LLM bound to the search tool."""
    search_tool = TavilySearch(
        max_results=MAX_ARTICLES,
        tavily_api_key=tavily_key,
        topic="news",
    )
    llm = ChatGroq(model=MODEL, api_key=groq_key, temperature=0.1)
    # Bind the tool so the LLM knows it can call it
    llm_with_tools = llm.bind_tools([search_tool])
    return llm_with_tools, search_tool


def run_agent(llm_with_tools, search_tool, query: str) -> list[dict]:
    """
    Agentic loop:
    - LLM decides to call the search tool
    - Tool result fed back to LLM
    - LLM evaluates and either searches again or produces final answer
    - Stops when LLM is satisfied or MAX_ITERATIONS reached
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f'Find the latest news about: "{query}"'),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"  [Iteration {iteration}] Thinking...", flush=True)

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # ── Check if LLM wants to call a tool ────────────────
        if response.tool_calls:
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                search_query = tool_args.get("query", query)

                print(f"  [Iteration {iteration}] Calling tool `{tool_name}` → \"{search_query}\"", flush=True)

                # Execute the tool
                try:
                    tool_result = search_tool.invoke(tool_args)
                    if isinstance(tool_result, str):
                        tool_content = tool_result
                    else:
                        tool_content = json.dumps(tool_result, default=str)
                except Exception as e:
                    tool_content = json.dumps({"error": str(e)})

                # Feed the result back into the conversation
                messages.append(
                    ToolMessage(
                        content=tool_content,
                        tool_call_id=tc["id"],
                    )
                )
        else:
            # LLM produced a final text answer — parse and return
            print(f"  [Iteration {iteration}] Agent satisfied, returning results.", flush=True)
            return _parse(response.content or "")

    # Exhausted iterations — try to parse whatever the last response was
    print(f"  [Max iterations reached] Returning best available results.", flush=True)
    last_text = next(
        (m.content for m in reversed(messages) if hasattr(m, "content") and isinstance(m.content, str)),
        ""
    )
    return _parse(last_text)


def _parse(text: str) -> list[dict]:
    """Strip markdown fences and parse the JSON array."""
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1:
        return [{"error": f"Could not parse response: {text[:200]}"}]
    try:
        data = json.loads(cleaned[start: end + 1])
        return data if isinstance(data, list) else [{"error": "Unexpected JSON structure."}]
    except json.JSONDecodeError as e:
        return [{"error": f"JSON parse error: {e}"}]


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

    if not groq_key or groq_key == "your_groq_api_key_here":
        sys.exit("Set GROQ_API_KEY in your .env file.")
    if not tavily_key or tavily_key == "your_tavily_api_key_here":
        sys.exit("Set TAVILY_API_KEY in your .env file.  Free key at https://app.tavily.com")

    llm_with_tools, search_tool = build_agent(groq_key, tavily_key)

    # Single query mode
    if "-q" in sys.argv:
        idx = sys.argv.index("-q")
        query = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ""
        if not query:
            sys.exit("Provide a query after -q")
        articles = run_agent(llm_with_tools, search_tool, query)
        print_results(articles, query)
        return

    # Interactive loop
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
