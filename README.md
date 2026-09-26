# Vera bot — magicpin AI Challenge

FastAPI service exposing `/v1/context`, `/v1/tick`, `/v1/reply`, `/v1/healthz`, `/v1/metadata` (+ optional `/v1/teardown`).

## Approach
1. **Context store**: versioned, idempotent per `(scope, context_id)`; higher version replaces atomically, lower → 409.
2. **Tick**: resolve trigger → merchant → category (→ customer); skip suppressed/expired/opted-out; sort by urgency; max one new thread per recipient per tick.
3. **Deterministic fact extraction**: digest item resolved from trigger ids, merchant numbers vs peer benchmarks, active offers, review themes, language preference, customer relationship.
4. **Per-trigger-kind strategy** (goal, compulsion lever, CTA type, what we deliver on "yes").
5. **LLM writer** (Gemini/OpenAI via env) → **strict validator**: no URLs, no taboo vocabulary, no numbers absent from context, no repeats, no qualifying questions after a "yes". One retry with the errors fed back, then a **deterministic template fallback** (bot works with no LLM at all).
6. **Replies**: rule-based intent router first — auto-reply (pattern + repetition, per merchant: flag once → wait 24h → end), opt-out/hostile → end, "no" → end, "later/kal" → wait, off-topic → polite redirect, "yes / haan / kar do / go ahead" → immediate action mode, second confirm → done + report-back.

## Env vars
| var | example |
|---|---|
| `LLM_PROVIDER` | `gemini` or `openai` |
| `LLM_API_KEY` | your key (set in Render dashboard, never commit) |
| `LLM_MODEL` | optional, default `gemini-2.5-flash` / `gpt-4o-mini` |
| `TEAM_NAME`, `CONTACT_EMAIL` | shown in `/v1/metadata` |

## Run / test
```
pip install -r requirements.txt
uvicorn bot:app --port 8080
python tests/smoke.py http://localhost:8080
```

## Tradeoffs / what would help
Deterministic fallback trades eloquence for guaranteed validity. More context that would help: real open slots per merchant, owner reply history, per-locality competitor data.
