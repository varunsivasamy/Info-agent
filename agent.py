"""
Core news agent logic — shared by CLI (news_agent.py) and WhatsApp bot (whatsapp_bot.py)
"""

import json
import logging
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

log = logging.getLogger(__name__)

MODEL          = "llama-3.3-70b-versatile"
MAX_ITERATIONS = 5
MAX_ARTICLES   = 5

SYSTEM_PROMPT = f"""You are a news retrieval agent. Today is {datetime.now().strftime("%d %B %Y")}.

Your job:
1. Use the `tavily_search` tool to search for real, recent news on the user's topic.
2. Evaluate the results — are they relevant, recent, and from credible sources?
3. If the results are poor or off-topic, refine your search query and search again.
4. Stop searching once you have {MAX_ARTICLES} or fewer good articles.

When satisfied, respond ONLY with a valid JSON array — no markdown, no extra text:
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
- Never fabricate headlines, dates, URLs, or facts.
- If nothing relevant found: [{{"error": "No relevant news found."}}]
"""


def build_agent(groq_key: str, tavily_key: str):
    """Build and return the LLM bound to the Tavily search tool."""
    search_tool = TavilySearch(
        max_results=MAX_ARTICLES,
        tavily_api_key=tavily_key,
        topic="news",
    )
    llm = ChatGroq(model=MODEL, api_key=groq_key, temperature=0.1)
    llm_with_tools = llm.bind_tools([search_tool])
    return llm_with_tools, search_tool


def run_agent(llm_with_tools, search_tool, query: str) -> list[dict]:
    """
    Agentic loop:
    - LLM decides when to call tavily_search
    - Tool result fed back to LLM
    - LLM evaluates and either searches again or returns final answer
    - Stops when satisfied or MAX_ITERATIONS reached
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f'Find the latest news about: "{query}"'),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        log.info("[Iteration %d] Invoking LLM...", iteration)
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            for tc in response.tool_calls:
                tool_args        = tc["args"]
                search_query     = tool_args.get("query", query)
                log.info("[Iteration %d] Tool call → \"%s\"", iteration, search_query)

                try:
                    tool_result  = search_tool.invoke(tool_args)
                    tool_content = (
                        tool_result if isinstance(tool_result, str)
                        else json.dumps(tool_result, default=str)
                    )
                except Exception as e:
                    tool_content = json.dumps({"error": str(e)})

                messages.append(
                    ToolMessage(content=tool_content, tool_call_id=tc["id"])
                )
        else:
            log.info("[Iteration %d] Agent satisfied, returning results.", iteration)
            return _parse(response.content or "")

    log.warning("Max iterations reached.")
    last_text = next(
        (m.content for m in reversed(messages)
         if hasattr(m, "content") and isinstance(m.content, str)),
        ""
    )
    return _parse(last_text)


def _parse(text: str) -> list[dict]:
    """Strip markdown fences and parse the JSON array from response."""
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1:
        return [{"error": f"Could not parse response: {text[:200]}"}]
    try:
        data = json.loads(cleaned[start: end + 1])
        return data if isinstance(data, list) else [{"error": "Unexpected JSON structure."}]
    except json.JSONDecodeError as e:
        return [{"error": f"JSON parse error: {e}"}]


def format_for_whatsapp(articles: list[dict], query: str) -> str:
    """Format article list as a clean WhatsApp text message."""
    if not articles:
        return "No news found for your query."

    if "error" in articles[0]:
        return f"Sorry, I couldn't find news on that topic: {articles[0]['error']}"

    lines = [f"📰 *News: {query}*\n"]
    for i, a in enumerate(articles, 1):
        headline  = a.get("headline", "N/A")
        source    = a.get("source", "?")
        published = a.get("published", "")
        summary   = a.get("summary", "")
        url       = a.get("url", "")

        lines.append(f"*{i}. {headline}*")
        lines.append(f"🗞 {source}  |  {published}")
        lines.append(f"{summary}")
        if url.startswith("http"):
            lines.append(f"🔗 {url}")
        lines.append("")   # blank line between articles

    return "\n".join(lines)
