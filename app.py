#!/usr/bin/env python3
"""
Entry point for Hugging Face Spaces deployment.
HF Spaces looks for app.py — this just re-exports the FastAPI app from whatsapp_bot.py
"""

from whatsapp_bot import app  # noqa: F401 — HF Spaces serves this
