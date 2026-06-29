#!/usr/bin/env python3
"""
WhatsApp News Agent — FastAPI webhook server
Receives WhatsApp messages via Meta Cloud API,
runs the news agent, and replies back.

Start server:
    uvicorn whatsapp_bot:app --host 0.0.0.0 --port 8000 --reload

Meta webhook config:
    Callback URL : https://<your-domain>/webhook
    Verify token : value of WHATSAPP_VERIFY_TOKEN in .env
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

# ── local agent ───────────────────────────────────────────────
from agent import build_agent, run_agent, format_for_whatsapp

# ── logging ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── config from .env ──────────────────────────────────────────
WHATSAPP_TOKEN    = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID   = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
VERIFY_TOKEN      = os.getenv("WHATSAPP_VERIFY_TOKEN", "news_agent_verify")
WHATSAPP_API_URL  = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"

# Build agent once at startup
GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
TAVILY_KEY = os.getenv("TAVILY_API_KEY", "")
llm_with_tools, search_tool = build_agent(GROQ_KEY, TAVILY_KEY)

app = FastAPI(title="WhatsApp News Agent")


# ── webhook verification (GET) ────────────────────────────────
@app.get("/webhook")
async def verify_webhook(request: Request):
    """
    Meta sends a GET request to verify the webhook URL.
    We must echo back hub.challenge if the verify token matches.
    """
    params = dict(request.query_params)
    mode      = params.get("hub.mode")
    token     = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        log.info("Webhook verified successfully.")
        return PlainTextResponse(challenge)

    log.warning("Webhook verification failed.")
    return Response(status_code=403)


# ── incoming messages (POST) ──────────────────────────────────
@app.post("/webhook")
async def receive_message(request: Request):
    """
    Meta sends a POST for every incoming WhatsApp message.
    We extract the text, run the agent, and reply.
    """
    body = await request.json()
    log.info("Incoming webhook: %s", json.dumps(body, indent=2))

    try:
        entry    = body["entry"][0]
        changes  = entry["changes"][0]
        value    = changes["value"]

        # Ignore status updates (delivered, read, etc.)
        if "messages" not in value:
            return {"status": "ignored"}

        message = value["messages"][0]
        from_number = message["from"]        # sender's WhatsApp number
        msg_type    = message.get("type")

        # Only handle text messages
        if msg_type != "text":
            await send_whatsapp_message(
                from_number,
                "Sorry, I can only handle text messages. Send me a news topic!"
            )
            return {"status": "non-text ignored"}

        user_text = message["text"]["body"].strip()
        log.info("Message from %s: %s", from_number, user_text)

        # Send a "searching..." acknowledgement first
        await send_whatsapp_message(from_number, f'🔍 Searching for news on: "{user_text}"...')

        # Run agent in background so we return 200 quickly to Meta
        asyncio.create_task(handle_query(from_number, user_text))

    except (KeyError, IndexError) as e:
        log.error("Failed to parse webhook body: %s", e)

    # Always return 200 to Meta immediately
    return {"status": "ok"}


async def handle_query(phone: str, query: str):
    """Run the news agent and send the result back to the user."""
    try:
        # run_agent is sync — run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        articles = await loop.run_in_executor(
            None, run_agent, llm_with_tools, search_tool, query
        )
        reply = format_for_whatsapp(articles, query)
    except Exception as e:
        log.error("Agent error: %s", e)
        reply = "Sorry, something went wrong while fetching news. Please try again."

    await send_whatsapp_message(phone, reply)


async def send_whatsapp_message(to: str, text: str):
    """Send a text message via Meta WhatsApp Cloud API."""
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
        resp = await client.post(WHATSAPP_API_URL, headers=headers, json=payload)
        if resp.status_code != 200:
            log.error("WhatsApp API error %s: %s", resp.status_code, resp.text)
        else:
            log.info("Message sent to %s", to)


# ── health check ──────────────────────────────────────────────
@app.get("/")
async def health():
    return {"status": "News Agent is running"}
