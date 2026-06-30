#!/usr/bin/env python3
"""
WhatsApp News Agent — FastAPI webhook server
Receives WhatsApp messages via Meta Cloud API,
runs the news agent, and replies with live news.
"""

import os
import json
import logging
import asyncio

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse

from agent import build_agent, run_agent, format_for_whatsapp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── lazy agent init — built on first message, not at startup ──
_llm_with_tools = None
_search_tool    = None

def get_agent():
    global _llm_with_tools, _search_tool
    if _llm_with_tools is None:
        groq_key   = os.environ.get("GROQ_API_KEY", "")
        tavily_key = os.environ.get("TAVILY_API_KEY", "")
        if not groq_key or not tavily_key:
            raise RuntimeError("GROQ_API_KEY and TAVILY_API_KEY must be set.")
        _llm_with_tools, _search_tool = build_agent(groq_key, tavily_key)
    return _llm_with_tools, _search_tool


app = FastAPI(title="WhatsApp News Agent")


@app.get("/")
async def health():
    return {"status": "News Agent is running"}


@app.get("/webhook")
async def verify_webhook(request: Request):
    """Meta webhook verification handshake."""
    verify_token = os.environ.get("WHATSAPP_VERIFY_TOKEN", "news_agent_verify")
    params    = dict(request.query_params)
    mode      = params.get("hub.mode")
    token     = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == verify_token:
        log.info("Webhook verified.")
        return PlainTextResponse(challenge)

    log.warning("Webhook verification failed.")
    return Response(status_code=403)


@app.post("/webhook")
async def receive_message(request: Request):
    """Handle incoming WhatsApp messages."""
    body = await request.json()
    log.info("Webhook POST received")

    try:
        value = body["entry"][0]["changes"][0]["value"]
        if "messages" not in value:
            return {"status": "ignored"}

        message     = value["messages"][0]
        from_number = message["from"]
        msg_type    = message.get("type")

        if msg_type != "text":
            await send_message(from_number, "I only handle text. Send me a news topic!")
            return {"status": "non-text ignored"}

        user_text = message["text"]["body"].strip()
        log.info("Message from %s: %s", from_number, user_text)

        await send_message(from_number, f'🔍 Searching news on: "{user_text}"...')
        asyncio.create_task(handle_query(from_number, user_text))

    except (KeyError, IndexError) as e:
        log.error("Parse error: %s", e)

    return {"status": "ok"}


async def handle_query(phone: str, query: str):
    """Run agent and send reply."""
    try:
        llm, tool = get_agent()
        loop      = asyncio.get_event_loop()
        articles  = await loop.run_in_executor(None, run_agent, llm, tool, query)
        reply     = format_for_whatsapp(articles, query)
    except Exception as e:
        log.error("Agent error: %s", e)
        reply = "Something went wrong fetching news. Please try again."
    await send_message(phone, reply)


async def send_message(to: str, text: str):
    """Send a WhatsApp message via Meta Cloud API."""
    phone_number_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
    token           = os.environ.get("WHATSAPP_TOKEN", "")
    url     = f"https://graph.facebook.com/v19.0/{phone_number_id}/messages"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
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
            log.info("Reply sent to %s", to)
