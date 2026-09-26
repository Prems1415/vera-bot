"""magicpin Vera challenge bot — HTTP layer + state.

Run:  uvicorn bot:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import composer
import llm

app = FastAPI(title="Vera bot")
START = time.time()
VERSION = "1.0.0"
SUBMITTED_AT = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

SCOPES = ("category", "merchant", "customer", "trigger")
contexts: dict[tuple[str, str], dict] = {}          # (scope, id) -> {"version", "payload"}
conversations: dict[str, dict] = {}                 # conv_id -> state
sent_suppression: dict[str, str] = {}               # suppression_key -> conv_id
merchant_inbound: dict[str, list[str]] = {}         # merchant_id -> recent inbound texts
auto_reply_count: dict[str, int] = {}               # merchant_id -> consecutive auto replies
merchant_optout: set[str] = set()
LOCK = asyncio.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get(scope: str, cid: str | None) -> dict | None:
    if not cid:
        return None
    v = contexts.get((scope, cid))
    return v["payload"] if v else None


def parse_dt(s: Any) -> datetime | None:
    if not s or not isinstance(s, str):
        return None
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def bad(reason: str, details: str = "", code: int = 400):
    return JSONResponse(status_code=code, content={"accepted": False, "reason": reason, "details": details})


# ------------------------------------------------------------------ health / metadata
@app.get("/")
async def root():
    return {"service": "vera-bot", "endpoints": ["/v1/context", "/v1/tick", "/v1/reply", "/v1/healthz", "/v1/metadata"]}


@app.get("/v1/healthz")
async def healthz():
    counts = {s: 0 for s in SCOPES}
    for (scope, _id) in list(contexts.keys()):
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    members = [m.strip() for m in os.getenv("TEAM_MEMBERS", os.getenv("TEAM_NAME", "Solo")).split(",") if m.strip()]
    return {
        "team_name": os.getenv("TEAM_NAME", "Vera Rebuilt"),
        "team_members": members,
        "model": llm.model_name(),
        "approach": ("Deterministic pipeline: context store with versioning -> trigger prioritisation + suppression -> "
                     "rule-based fact extraction (digest resolution, peer gaps, offers, language) -> per-trigger-kind strategy -> "
                     "LLM writer with strict validation (no URLs, no taboo words, no numbers absent from context, no repeats) -> "
                     "one retry -> deterministic template fallback. Replies: rule-based intent router (auto-reply, opt-out, accept, "
                     "defer, off-topic, question) before any LLM call."),
        "contact_email": os.getenv("CONTACT_EMAIL", ""),
        "version": VERSION,
        "submitted_at": os.getenv("SUBMITTED_AT", SUBMITTED_AT),
    }


# ------------------------------------------------------------------ context
@app.post("/v1/context")
async def push_context(request: Request):
    try:
        body = await request.json()
    except Exception:
        return bad("malformed_json")
    if not isinstance(body, dict):
        return bad("malformed_body")
    scope, cid, version, payload = body.get("scope"), body.get("context_id"), body.get("version"), body.get("payload")
    if scope not in SCOPES:
        return bad("invalid_scope", f"scope must be one of {SCOPES}")
    if not isinstance(cid, str) or not cid:
        return bad("invalid_context_id")
    try:
        version = int(version)
    except Exception:
        return bad("invalid_version")
    if not isinstance(payload, dict):
        return bad("invalid_payload", "payload must be an object")
    key = (scope, cid)
    async with LOCK:
        cur = contexts.get(key)
        if cur and cur["version"] > version:
            return JSONResponse(status_code=409, content={"accepted": False, "reason": "stale_version", "current_version": cur["version"]})
        if cur and cur["version"] == version:
            return {"accepted": True, "ack_id": f"ack_{cid}_v{version}", "stored_at": cur["stored_at"], "note": "duplicate_version_noop"}
        stored = now_iso()
        contexts[key] = {"version": version, "payload": payload, "stored_at": stored}
    return {"accepted": True, "ack_id": f"ack_{cid}_v{version}", "stored_at": stored}


@app.post("/v1/teardown")
async def teardown():
    async with LOCK:
        contexts.clear(); conversations.clear(); sent_suppression.clear()
        merchant_inbound.clear(); auto_reply_count.clear(); merchant_optout.clear()
    return {"ok": True}


# ------------------------------------------------------------------ tick
def _short(mid: str) -> str:
    parts = mid.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else mid


def _new_conv_id(merchant_id: str, trigger: dict, customer_id: str | None) -> str:
    base = f"conv_{_short(merchant_id)}_{trigger.get('kind', 'msg')}"
    if customer_id:
        base += "_" + customer_id.split("_")[1] if len(customer_id.split("_")) > 1 else ""
    cid, n = base, 1
    while cid in conversations:
        n += 1
        cid = f"{base}_{n}"
    return cid


async def _compose_safe(category, merchant, trigger, customer, prior) -> dict:
    try:
        return await asyncio.wait_for(composer.compose(category, merchant, trigger, customer, prior), timeout=12.0)
    except Exception as e:
        print(f"[tick] compose fallback: {type(e).__name__}: {e}")
        # deterministic path (no LLM)
        send_as = "merchant_on_behalf" if (trigger.get("scope") == "customer" and customer) else "vera"
        facts = composer.build_facts(category, merchant, trigger, customer)
        hing = composer.wants_hinglish(merchant, customer if send_as == "merchant_on_behalf" else None)
        body, cta = composer.fallback_compose(facts, send_as, hing)
        return {"body": body, "cta": cta, "send_as": send_as, "facts": facts,
                "rationale": f"{trigger.get('kind')} trigger; deterministic template (LLM timed out) [composer=fallback]"}


@app.post("/v1/tick")
async def tick(request: Request):
    try:
        body = await request.json()
    except Exception:
        return {"actions": []}
    now = parse_dt((body or {}).get("now")) or datetime.now(timezone.utc)
    ids = (body or {}).get("available_triggers") or []
    if not isinstance(ids, list):
        return {"actions": []}

    candidates = []
    for tid in ids:
        trg = get("trigger", tid) if isinstance(tid, str) else None
        if not trg:
            continue
        skey = trg.get("suppression_key") or f"trg:{tid}"
        if skey in sent_suppression:
            continue
        mid = trg.get("merchant_id") or (trg.get("payload") or {}).get("merchant_id")
        merchant = get("merchant", mid)
        if not merchant or mid in merchant_optout:
            continue
        exp = parse_dt(trg.get("expires_at"))
        # skip truly expired triggers; tolerate a clock far past the dataset timeline (simulator uses wall-clock)
        if exp and exp < now and (now - exp).days < 30:
            continue
        if (merchant.get("subscription") or {}).get("status") == "expired" and trg.get("kind") not in (
                "winback_eligible", "dormant_with_vera", "renewal_due"):
            pass  # still allowed, just lower priority
        candidates.append((int(trg.get("urgency") or 1), tid, trg, merchant))

    candidates.sort(key=lambda x: -x[0])
    chosen, used = [], set()
    for urg, tid, trg, merchant in candidates:
        cust_id = trg.get("customer_id") or (trg.get("payload") or {}).get("customer_id")
        slot = ("c", cust_id) if (trg.get("scope") == "customer" and cust_id) else ("m", merchant.get("merchant_id"))
        if slot in used:
            continue  # restraint: one new thread per recipient per tick
        used.add(slot)
        chosen.append((tid, trg, merchant, cust_id))
        if len(chosen) >= 20:
            break

    sem = asyncio.Semaphore(8)

    async def build(tid, trg, merchant, cust_id):
        async with sem:
            category = get("category", merchant.get("category_slug"))
            customer = get("customer", cust_id)
            res = await _compose_safe(category, merchant, trg, customer, [])
            mid = merchant.get("merchant_id") or trg.get("merchant_id")
            send_as = res["send_as"]
            conv_id = _new_conv_id(mid, trg, cust_id if send_as == "merchant_on_behalf" else None)
            conversations[conv_id] = {
                "merchant_id": mid, "customer_id": cust_id if send_as == "merchant_on_behalf" else None,
                "trigger_id": tid, "send_as": send_as, "facts": res["facts"], "state": "open",
                "hinglish": composer.wants_hinglish(merchant, customer if send_as == "merchant_on_behalf" else None),
                "turns": [{"from": "vera", "body": res["body"]}], "bodies": [res["body"]], "accepts": 0,
                "from_role": "customer" if send_as == "merchant_on_behalf" else "merchant",
            }
            sent_suppression[trg.get("suppression_key") or f"trg:{tid}"] = conv_id
            facts = res["facts"]
            first_param = (facts.get("customer") or {}).get("name") if send_as == "merchant_on_behalf" else facts["merchant"]["salutation"]
            kind = trg.get("kind", "update")
            return {
                "conversation_id": conv_id,
                "merchant_id": mid,
                "customer_id": cust_id if send_as == "merchant_on_behalf" else None,
                "send_as": send_as,
                "trigger_id": tid,
                "template_name": f"{'merchant' if send_as == 'merchant_on_behalf' else 'vera'}_{kind}_v1",
                "template_params": [str(first_param or ""), composer.humanize(kind), res["body"][:120]],
                "body": res["body"],
                "cta": res["cta"],
                "suppression_key": trg.get("suppression_key") or f"trg:{tid}",
                "rationale": res["rationale"],
            }

    try:
        actions = await asyncio.wait_for(asyncio.gather(*(build(*c) for c in chosen), return_exceptions=True), timeout=13.5)
    except asyncio.TimeoutError:
        actions = []
    actions = [a for a in actions if isinstance(a, dict) and a.get("body")]
    return {"actions": actions}


# ------------------------------------------------------------------ reply
def _wait_seconds(msg: str) -> int:
    low = msg.lower()
    if re.search(r"next week|agle hafte", low):
        return 3 * 86400
    if re.search(r"tomorrow|\bkal\b", low):
        return 86400
    m = re.search(r"(\d+)\s*(min|minute)", low)
    if m:
        return max(300, int(m.group(1)) * 60)
    m = re.search(r"(\d+)\s*(hr|hour|ghante)", low)
    if m:
        return int(m.group(1)) * 3600
    return 7200


@app.post("/v1/reply")
async def reply(request: Request):
    try:
        body = await request.json()
    except Exception:
        return {"action": "wait", "wait_seconds": 3600, "rationale": "Malformed reply payload; backing off."}
    if not isinstance(body, dict):
        return {"action": "wait", "wait_seconds": 3600, "rationale": "Malformed reply payload; backing off."}
    conv_id = str(body.get("conversation_id") or "conv_unknown")
    mid = body.get("merchant_id")
    cust_id = body.get("customer_id")
    role = body.get("from_role") or "merchant"
    msg = str(body.get("message") or "")

    conv = conversations.get(conv_id)
    if conv is None:
        # conversation we didn't start (or state lost on restart): rebuild from stored contexts
        merchant = get("merchant", mid) or {}
        category = get("category", merchant.get("category_slug")) if merchant else None
        customer = get("customer", cust_id)
        facts = composer.build_facts(category, merchant, {}, customer) if merchant else None
        conv = {"merchant_id": mid, "customer_id": cust_id, "trigger_id": None, "facts": facts, "state": "open",
                "send_as": "merchant_on_behalf" if role == "customer" else "vera",
                "hinglish": composer.wants_hinglish(merchant, customer) if merchant else False,
                "turns": [], "bodies": [], "accepts": 0, "from_role": role}
        conversations[conv_id] = conv
    conv["from_role"] = role
    mid = mid or conv.get("merchant_id")

    if conv.get("state") == "ended":
        return {"action": "end", "rationale": "Conversation already closed; not re-engaging."}

    key = f"{role}:{cust_id or mid}"
    prior_in = [t["body"] for t in conv["turns"] if t["from"] != "vera"] + merchant_inbound.get(key, [])[-5:]
    intent = composer.classify(msg, prior_in)
    conv["turns"].append({"from": role, "body": msg})
    merchant_inbound.setdefault(key, []).append(msg)
    merchant_inbound[key] = merchant_inbound[key][-10:]

    if intent != "auto_reply":
        auto_reply_count[key] = 0

    if intent == "auto_reply":
        n = auto_reply_count.get(key, 0) + 1
        auto_reply_count[key] = n
        if n == 1:
            text = ("Lagta hai yeh auto-reply hai 🙂 Owner/manager jab dekhein, bas 'YES' reply kar dein — main baaki sab ready rakhungi."
                    if conv.get("hinglish") else
                    "Looks like an auto-reply 🙂 Whenever the owner sees this, just reply YES and I'll have everything ready.")
            if text in conv["bodies"]:
                return {"action": "wait", "wait_seconds": 86400, "rationale": "Repeated auto-reply; waiting 24h for the owner."}
            conv["bodies"].append(text); conv["turns"].append({"from": "vera", "body": text})
            return {"action": "send", "body": text, "cta": "binary_yes_no",
                    "rationale": "Detected WhatsApp Business auto-reply (canned phrasing); one short owner-flag prompt, no pitch."}
        if n == 2:
            return {"action": "wait", "wait_seconds": 86400,
                    "rationale": "Same canned auto-reply again → owner not at the phone. Backing off 24h instead of burning turns."}
        conv["state"] = "ended"
        return {"action": "end", "rationale": f"Auto-reply received {n}x in a row; exiting gracefully to avoid spamming."}

    if intent == "opt_out":
        conv["state"] = "ended"
        if mid and role == "merchant":
            merchant_optout.add(mid)
        return {"action": "end", "rationale": "Merchant asked to stop / was hostile; ending immediately and suppressing further outreach."}

    if intent == "decline":
        conv["state"] = "ended"
        return {"action": "end", "rationale": "Clear 'no'; respecting it and closing the conversation politely."}

    if intent == "defer":
        return {"action": "wait", "wait_seconds": _wait_seconds(msg), "rationale": "Merchant asked for time; backing off accordingly."}

    if intent == "empty":
        return {"action": "wait", "wait_seconds": 3600, "rationale": "Empty message; waiting."}

    vera_sends = sum(1 for t in conv["turns"] if t["from"] == "vera")
    if vera_sends >= 6:
        conv["state"] = "ended"
        return {"action": "end", "rationale": "Conversation has run its course (6 sends); closing to avoid fatigue."}

    if intent == "accept":
        conv["accepts"] = conv.get("accepts", 0) + 1
        if conv["accepts"] >= 2 and not conv.get("confirmed_sent"):
            conv["confirmed_sent"] = True

    try:
        res = await asyncio.wait_for(composer.respond(conv, conv.get("facts"), msg, intent, conv["bodies"]), timeout=12.5)
    except Exception as e:
        print(f"[reply] fallback: {type(e).__name__}: {e}")
        kind = composer._g(conv.get("facts"), "trigger", "kind")
        deliverable = composer.STRATEGY.get(kind, composer.DEFAULT_STRATEGY)[3] if kind else "the draft"
        b, cta = composer.fallback_reply(intent, conv.get("facts"), deliverable, conv.get("hinglish", False), msg, conv.get("accepts", 1))
        res = {"action": "send", "body": b, "cta": cta, "rationale": f"intent={intent}; deterministic reply [composer=fallback]"}

    if res["body"] in conv["bodies"]:
        res["body"] = res["body"] + (" (Reply CONFIRM whenever ready.)" if "CONFIRM" not in res["body"] else " 🙂")
    conv["bodies"].append(res["body"])
    conv["turns"].append({"from": "vera", "body": res["body"]})
    return res


# ------------------------------------------------------------------ keep-alive (Render free tier sleeps after 15 min idle)
@app.on_event("startup")
async def _keepalive():
    url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("SELF_URL")
    if not url:
        return

    async def loop():
        async with httpx.AsyncClient(timeout=10) as c:
            while True:
                await asyncio.sleep(600)
                try:
                    await c.get(url.rstrip("/") + "/v1/healthz")
                except Exception:
                    pass

    asyncio.create_task(loop())


@app.get("/v1/llmcheck")
async def llmcheck():
    """Diagnostic: confirms the LLM key works (never returns the key)."""
    if not llm.enabled():
        return {"llm_enabled": False, "provider": llm.PROVIDER}
    t = time.time()
    out = await llm.complete_json('Return JSON {"ok": true, "word": "<one word>"}', "Say hello in one word.", timeout=10)
    return {"llm_enabled": True, "provider": llm.PROVIDER, "model": llm.MODEL, "ok": bool(out), "latency_s": round(time.time() - t, 2)}
