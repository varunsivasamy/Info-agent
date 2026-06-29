---
title: News Agent
emoji: 📰
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 📰 News Agent — WhatsApp Bot

An AI-powered news retrieval agent deployed as a WhatsApp bot.

Built with **LangChain + Groq (LLaMA 3.3 70B) + Tavily** — real agentic loop that:
1. Receives your WhatsApp message
2. LLM decides what to search for and calls the Tavily web search tool
3. Evaluates results — retries with a refined query if needed
4. Replies with real, live news articles

## Stack
- **Groq** — LLaMA 3.3 70B (fast, free)
- **Tavily** — real-time web search for AI agents
- **LangChain** — agentic tool-calling loop
- **FastAPI** — webhook server for Meta WhatsApp API

## Environment Variables (set in HF Spaces Secrets)

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | From [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | From [app.tavily.com](https://app.tavily.com) |
| `WHATSAPP_TOKEN` | Meta WhatsApp Cloud API token |
| `WHATSAPP_PHONE_NUMBER_ID` | From Meta Developer dashboard |
| `WHATSAPP_VERIFY_TOKEN` | Any string you choose, e.g. `news_agent_verify` |

## Local Development
```bash
# Create virtual environment
python -m venv newsagent
newsagent\Scripts\activate      # Windows
source newsagent/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Copy and fill in your keys
cp .env.example .env

# Run CLI
python news_agent.py -q "AI news today"

# Run WhatsApp server locally
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

## Webhook Setup (Meta Developer Console)
1. Deploy this Space
2. Go to [developers.facebook.com](https://developers.facebook.com) → your WhatsApp app → Webhooks
3. Set Callback URL: `https://<your-space-url>/webhook`
4. Set Verify Token: same value as `WHATSAPP_VERIFY_TOKEN`
5. Subscribe to the `messages` field
