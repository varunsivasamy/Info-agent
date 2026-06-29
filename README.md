# Intelligent News Agent — Terminal

A command-line news retrieval and verification agent powered by **Google Gemini 2.0 Flash** with live Google Search grounding.

## Features

- Live news via Gemini's Google Search grounding tool
- Verification & retry logic (up to 3 retries)
- Deduplication of same-event stories
- Rich terminal output with colour formatting
- Interactive REPL mode or single-query CLI mode
- Optional raw `--json` output for scripting

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a Gemini API key

Free at https://aistudio.google.com/apikey

### 3. Set your API key

```bash
export GEMINI_API_KEY="your_key_here"
```

Or add it to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.).

## Usage

### Interactive mode (recommended)

```bash
python news_agent.py
```

Type any topic at the `News ›` prompt. Type `help` for examples, `quit` to exit.

### Single-query mode

```bash
python news_agent.py -q "AI news today"
python news_agent.py -q "cricket scores latest"
python news_agent.py -q "Tesla news this week"
```

### Raw JSON output (for scripting / piping)

```bash
python news_agent.py -q "stock market today" --json
python news_agent.py -q "technology news" --json | jq '.[].headline'
```

### Pass API key inline

```bash
python news_agent.py --api-key YOUR_KEY -q "world news today"
```

## Options

```
  -q, --query TEXT     Run a single query and exit
  --json               Print raw JSON instead of formatted output
  --api-key TEXT       Gemini API key (or set GEMINI_API_KEY env var)
  -h, --help           Show help
```

## Example queries

| Query | What it finds |
|-------|--------------|
| `AI news today` | Latest AI / ML news |
| `cricket scores yesterday` | Recent match results |
| `stock market today` | Finance & markets |
| `Tesla latest news` | Company-specific news |
| `technology updates 2026` | General tech news |
| `global politics today` | World news |
| `startup funding this week` | VC & startup news |
| `health science news` | Medical updates |

## Output format

Each article shows:
- Publisher name + publication date
- Headline
- 3–5 sentence factual summary
- Article URL

## JSON schema (for `--json` mode)

```json
[
  {
    "headline": "Full article headline",
    "published": "29 June 2026, 14:30 UTC",
    "source": "Reuters",
    "summary": "3–5 sentence factual summary.",
    "url": "https://reuters.com/...",
    "verified": true,
    "retries": 0
  }
]
```

## Configuration

Edit the constants at the top of `news_agent.py`:

```python
GEMINI_MODEL       = "gemini-2.0-flash"   # or "gemini-1.5-pro" for higher quality
MAX_RETRIES        = 3                     # verification retries
ARTICLES_PER_QUERY = 5                     # max articles returned
```
