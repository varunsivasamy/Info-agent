#!/usr/bin/env python3
"""
WhatsApp News Agent — FastAPI webhook server
Receives WhatsApp messages via Meta Cloud API,
runs the news agent, and replies back.

Start server:
    uvicorn whatsapp_bot:app --host 0.0.0.0 --port 7860 --reload

Meta webhook config:
    Callback URL : https://<your-space-url>/webhook
    Verify token : value of WHATSAPP_VERIFY_TOKEN in Secrets
"""

import os
import json
import logging
import asyncio
from dotenv import load_dotenv

load_dotenv()

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse

from agent import build_agent, run_agent, format_for_whatsapp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────
WHATSAPP_TOKEN  = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
VERIFY_TOKEN    = os.getenv("WHATSAPP_VERIFY_TOKEN", "news_agent_verify")
GROQ_KEY        = os.getenv("GROQ_API_KEY", "")
TAVILY_KEY      = os.getenv("TAVILY_API_KEY", "")

# ── lazy agent init — built on first request, not at import ──
_llm_with_tools = None
_search_tool    = None

def get_agent():
    global _llm_with_tools, _search_tool
    if _llm_with_tools is None:
        if not GROQ_KEY or not TAVILY_KEY:
            raise RuntimeError("GROQ_API_KEY and TAVILY_API_KEY must be set in HF Secrets.")
        log.info("Initialising agent...")
        _llm_with_tools, _search_tool = build_agent(GROQ_KEY, TAVILY_KEY)
        log.info("Agent ready.")
    return _llm_with_tools, _search_tool

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(title="WhatsApp News Agent")


@app.get("/")
async def health():
    return {"status": "News Agent is running"}


@app.get("/webhook")
async def verify_webhook(request: Request):
    """Meta webhook verification handshake."""
    params    = dict(request.query_params)
    mode      = params.get("hub.mode")
    token     = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        log.info("Webhook verified.")
        return PlainTextResponse(challenge)

    log.warning("Webhook verification failed.")
    return Response(status_code=403)


@app.post("/webhook")
async def receive_message(request: Request):
    """Handle incoming WhatsApp messages."""
    body = await request.json()
    log.info("Webhook received: %s", json.dumps(body, indent=2))

    try:
        value = body["entry"][0]["changes"][0]["value"]

        if "messages" not in value:
            return {"status": "ignored"}

        message     = value["messages"][0]
        from_number = message["from"]
        msg_type    = message.get("type")

        if msg_type != "text":
            await send_message(from_number, "I only handle text messages. Send me a news topic!")
            return {"status": "non-text ignored"}

        user_text = message["text"]["body"].strip()
        log.info("Message from %s: %s", from_number, user_text)

        await send_message(from_number, f'🔍 Searching for news on: "{user_text}"...')
        asyncio.create_task(handle_query(from_number, user_text))

    except (KeyError, IndexError) as e:
        log.error("Parse error: %s", e)

    return {"status": "ok"}


async def handle_query(phone: str, query: str):
    """Run the agent in a thread pool and send the reply."""
    try:
        llm, tool = get_agent()
        loop      = asyncio.get_event_loop()
        articles  = await loop.run_in_executor(None, run_agent, llm, tool, query)
        reply     = format_for_whatsapp(articles, query)
    except Exception as e:
        log.error("Agent error: %s", e)
        reply = "Something went wrong while fetching news. Please try again."

    await send_message(phone, reply)


async def send_message(to: str, text: str):
    """Send a WhatsApp message via Meta Cloud API."""
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            log.error("WhatsApp API error %s: %s", resp.status_code, resp.text)
        else:
            log.info("Sent reply to %s", to)
