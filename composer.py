"""Deterministic pipeline around an LLM writer.

compose():  contexts -> facts -> strategy -> LLM draft -> validate -> (retry) -> fallback
respond():  merchant reply -> intent classification (rules) -> action (end/wait/send) -> body
"""
from __future__ import annotations

import json
import re
from typing import Any

import llm

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
CTA_VALUES = {"open_ended", "binary_yes_no", "binary_confirm_cancel", "multi_choice_slot", "none"}

URL_RE = re.compile(r"(https?://|www\.)\S+|\b[\w-]+\.(com|in|org|net|io|co)\b(/\S*)?", re.I)

# Strategy per trigger kind: (goal, lever, default CTA, deliverable offered on "yes")
STRATEGY: dict[str, tuple[str, str, str, str]] = {
    "research_digest": ("share the digest item as a peer; tie it to this merchant's patient/customer cohort", "curiosity + reciprocity + source citation", "open_ended", "the 2-min summary plus a patient-ed WhatsApp draft"),
    "regulation_change": ("flag the compliance change, the deadline and exactly what to check", "loss aversion + deadline", "binary_yes_no", "a 5-point compliance checklist for your setup"),
    "supply_alert": ("urgent: name molecule, batches and manufacturer; offer to identify affected customers", "urgency + effort externalization", "binary_yes_no", "the list of affected customers plus a ready-to-send customer note"),
    "cde_opportunity": ("share the CDE/training event: date, credits, fee", "curiosity + low effort", "binary_yes_no", "the registration details and a calendar reminder"),
    "recall_due": ("remind the customer their recall is due; offer the concrete slots", "convenience + specificity", "multi_choice_slot", "the booking confirmation"),
    "chronic_refill_due": ("remind the customer their medicines run out soon; offer delivery", "convenience + loss aversion", "binary_yes_no", "the refill order for delivery"),
    "customer_lapsed_hard": ("warm win-back to the lapsed customer referencing their past goal; no guilt", "reciprocity + easy restart", "binary_yes_no", "a restart slot booking"),
    "customer_lapsed_soft": ("gentle nudge to the customer to come back", "convenience", "binary_yes_no", "a slot booking"),
    "trial_followup": ("follow up on the trial; offer the next session", "momentum + specificity", "multi_choice_slot", "the next session booking"),
    "wedding_package_followup": ("bridal follow-up: countdown to the wedding and the next prep step", "timeline urgency", "binary_yes_no", "the prep-program booking"),
    "appointment_tomorrow": ("confirm tomorrow's appointment", "convenience", "binary_confirm_cancel", "the confirmation"),
    "perf_dip": ("name the metric drop with numbers; give one likely cause and one fix", "loss aversion + effort externalization", "binary_yes_no", "a fresh Google post plus an offer refresh, ready for your OK"),
    "seasonal_perf_dip": ("reframe the dip as expected seasonality (don't panic); suggest one smart move for the lull", "judgment + reassurance", "binary_yes_no", "a retention push drafted for your current members"),
    "perf_spike": ("celebrate the spike with numbers and the likely driver; suggest how to double down", "momentum + curiosity", "binary_yes_no", "a follow-up post to ride the momentum"),
    "milestone_reached": ("call out the milestone (or how close it is); suggest a way to cross/celebrate it", "progress + social proof", "binary_yes_no", "a review-request message for recent happy customers"),
    "review_theme_emerged": ("surface the recurring review theme with the real quote; suggest one operational fix and a reply", "loss aversion + specificity", "binary_yes_no", "polite public replies to those reviews plus one fix note"),
    "renewal_due": ("renewal reminder: days left, plan, amount; show value from their own numbers", "loss aversion", "binary_yes_no", "the renewal set up (no auto-charge)"),
    "winback_eligible": ("win back an expired merchant with what they've lost since expiry (their numbers)", "loss aversion", "binary_yes_no", "a reactivation with your old listing restored"),
    "dormant_with_vera": ("re-open the conversation with one useful fresh insight from their data; no guilt", "curiosity + reciprocity", "open_ended", "a quick profile audit summary"),
    "festival_upcoming": ("festival planning: date, days to go, a category-fit offer idea from the catalog", "timeliness + effort externalization", "binary_yes_no", "a festival offer + Google post draft"),
    "ipl_match_today": ("match-day play: match, time, venue; give a data-informed recommendation (weeknight vs weekend matters)", "timeliness + judgment", "binary_yes_no", "a match-night banner plus a story post"),
    "category_seasonal": ("seasonal demand shift with the trend numbers; one shelf/offer action", "specificity + timeliness", "binary_yes_no", "a shelf checklist and a customer broadcast draft"),
    "category_trend_movement": ("trend signal with numbers; how this merchant can capture it", "curiosity", "binary_yes_no", "a post targeting that search"),
    "gbp_unverified": ("Google profile unverified: the uplift they are missing and the verification path", "loss aversion + effort externalization", "binary_yes_no", "the verification started for you"),
    "competitor_opened": ("new competitor nearby (name, distance, their offer); suggest a non-price-war response", "loss aversion + judgment", "binary_yes_no", "a differentiation post plus an offer tweak"),
    "active_planning_intent": ("the merchant already asked for this — give a concrete draft plan now (structure, price from catalog, timing), not questions", "effort externalization", "binary_confirm_cancel", "the full plan set up and published"),
    "curious_ask_due": ("ask the merchant one easy, specific question about their business this week; promise something useful back", "asking the merchant + reciprocity", "open_ended", "a post built around their answer"),
    "scheduled_recurring": ("light weekly check-in with one data point", "curiosity", "open_ended", "a short weekly summary"),
    "weather_heatwave": ("weather event today; category-fit adjustment", "timeliness", "binary_yes_no", "a weather-timed post"),
    "local_news_event": ("local event today; what it means for footfall", "timeliness", "binary_yes_no", "a timely post"),
}
DEFAULT_STRATEGY = ("explain why this matters to the merchant now using their numbers; one clear next step", "specificity + effort externalization", "binary_yes_no", "the draft ready for your OK")

TONE_HINT = {
    "dentists": "clinical peer (colleague-to-colleague). Address as 'Dr. <first name>'. Technical terms OK. No hype, no emojis except maybe one.",
    "salons": "warm, practical, friendly expert. Service+price framing (e.g. 'Hair Spa @ ₹499').",
    "restaurants": "fellow operator, busy and practical. Use covers/footfall/AOV vocabulary naturally.",
    "gyms": "energetic but disciplined coach tone. No body-shaming, no guaranteed results.",
    "pharmacies": "trustworthy, precise neighbourhood pharmacist. Exact molecules/batches. No medical claims.",
}

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _g(d: Any, *path, default=None):
    for p in path:
        if isinstance(d, dict):
            d = d.get(p)
        else:
            return default
    return default if d is None else d


def wants_hinglish(merchant: dict | None, customer: dict | None = None) -> bool:
    if customer:
        pref = str(_g(customer, "identity", "language_pref", default="")).lower()
        if pref:
            return "hi" in pref or "hindi" in pref
    langs = [str(x).lower() for x in (_g(merchant, "identity", "languages", default=[]) or [])]
    return "hi" in langs


def salutation(merchant: dict, category_slug: str) -> str:
    owner = _g(merchant, "identity", "owner_first_name")
    name = _g(merchant, "identity", "name", default="there")
    if owner:
        return f"Dr. {owner}" if category_slug == "dentists" else owner
    return name


def find_digest_item(category: dict | None, trigger: dict) -> dict | None:
    if not category:
        return None
    items = {d.get("id"): d for d in category.get("digest", []) or [] if isinstance(d, dict)}
    payload = trigger.get("payload") or {}
    for v in payload.values():
        if isinstance(v, str) and v in items:
            return items[v]
        if isinstance(v, dict) and v.get("title"):
            return v  # inline top_item
    return None


PLACE = {"dentists": "clinic", "pharmacies": "store", "salons": "salon", "gyms": "studio", "restaurants": "outlet"}


def humanize(s: Any) -> str:
    return str(s).replace("_", " ")


# --------------------------------------------------------------------------------------
# Fact extraction (deterministic)
# --------------------------------------------------------------------------------------

def build_facts(category: dict | None, merchant: dict, trigger: dict, customer: dict | None) -> dict:
    slug = merchant.get("category_slug") or _g(category, "slug", default="")
    perf = merchant.get("performance") or {}
    peer = (category or {}).get("peer_stats") or {}
    facts: dict[str, Any] = {
        "category": slug,
        "merchant": {
            "name": _g(merchant, "identity", "name"),
            "salutation": salutation(merchant, slug),
            "city": _g(merchant, "identity", "city"),
            "locality": _g(merchant, "identity", "locality"),
            "verified": _g(merchant, "identity", "verified"),
            "languages": _g(merchant, "identity", "languages", default=[]),
            "subscription": merchant.get("subscription"),
            "performance_30d": perf,
            "active_offers": [o.get("title") for o in merchant.get("offers", []) or [] if o.get("status") == "active"],
            "expired_offers": [o.get("title") for o in merchant.get("offers", []) or [] if o.get("status") == "expired"],
            "signals": merchant.get("signals", []),
            "customer_aggregate": merchant.get("customer_aggregate"),
            "review_themes": merchant.get("review_themes"),
            "recent_conversation": (merchant.get("conversation_history") or [])[-4:],
        },
        "peer_benchmarks": peer,
        "trigger": {
            "kind": trigger.get("kind"),
            "scope": trigger.get("scope"),
            "source": trigger.get("source"),
            "urgency": trigger.get("urgency"),
            "payload": trigger.get("payload"),
            "expires_at": trigger.get("expires_at"),
        },
    }
    item = find_digest_item(category, trigger)
    if item:
        facts["trigger"]["resolved_digest_item"] = item
    if category:
        facts["category_catalog_offers"] = [o.get("title") for o in category.get("offer_catalog", [])][:8]
        facts["category_voice"] = {
            "tone": _g(category, "voice", "tone"),
            "vocab_allowed": (_g(category, "voice", "vocab_allowed", default=[]) or [])[:12],
            "taboo": _g(category, "voice", "vocab_taboo", default=[]),
        }
        facts["seasonal_beats"] = category.get("seasonal_beats", [])[:4]
        facts["trend_signals"] = category.get("trend_signals", [])[:4]
        # a few other fresh digest items (so newly-pushed context is visible), compact
        others = [d for d in category.get("digest", []) or [] if d is not item][:3]
        facts["other_digest_items"] = [{k: d.get(k) for k in ("title", "source", "summary") if d.get(k)} for d in others]
    if customer:
        facts["customer"] = {
            "name": _g(customer, "identity", "name"),
            "language_pref": _g(customer, "identity", "language_pref"),
            "relationship": customer.get("relationship"),
            "state": customer.get("state"),
            "preferences": customer.get("preferences"),
            "consent_scope": _g(customer, "consent", "scope"),
        }
    return facts


def allowed_numbers(facts: dict) -> set[str]:
    text = json.dumps(facts, ensure_ascii=False)
    nums: set[str] = set()
    for m in re.finditer(r"-?\d[\d,]*(?:\.\d+)?", text):
        raw = m.group().replace(",", "").lstrip("-")
        _add_num(nums, raw)
        try:
            v = float(raw)
            if 0 < v < 1:  # ratios -> percentages
                for pv in (v * 100,):
                    _add_num(nums, f"{pv:.1f}")
                    _add_num(nums, str(round(pv)))
        except ValueError:
            pass
    # simple derived values: peer gaps etc. are allowed if both operands exist -> too complex, keep lenient below
    return nums


def _add_num(s: set[str], raw: str):
    try:
        v = float(raw)
    except ValueError:
        return
    s.add(raw)
    if v == int(v):
        s.add(str(int(v)))
    else:
        s.add(f"{v:.1f}".rstrip("0").rstrip("."))


SAFE_SMALL = {str(i) for i in range(0, 13)} | {"15", "20", "24", "30", "45", "48", "60", "90", "100"}


def unsupported_numbers(body: str, facts: dict) -> list[str]:
    allowed = allowed_numbers(facts)
    bad = []
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", body):
        raw = m.group().replace(",", "")
        norm = raw
        try:
            v = float(raw)
            norm = str(int(v)) if v == int(v) else f"{v:.1f}".rstrip("0").rstrip(".")
        except ValueError:
            pass
        if raw in allowed or norm in allowed or norm in SAFE_SMALL:
            continue
        if re.fullmatch(r"20[2-3]\d", norm):
            continue
        bad.append(raw)
    return bad


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
QUALIFYING = ["would you", "do you", "can you tell", "what if", "how about"]


def validate_body(body: Any, facts: dict, prior_bodies: list[str], mode: str = "initial") -> list[str]:
    errs = []
    if not isinstance(body, str) or not body.strip():
        return ["empty body"]
    b = body.strip()
    if len(b) > 700:
        errs.append("too long (keep under ~600 chars)")
    if URL_RE.search(b):
        errs.append("contains a URL/domain — not allowed")
    taboo = [t for t in (_g(facts, "category_voice", "taboo", default=[]) or [])]
    for t in taboo:
        core = re.sub(r"\(.*?\)", "", str(t)).strip().lower()
        if core and core in b.lower():
            errs.append(f"uses taboo phrase '{core}'")
    bad = unsupported_numbers(b, facts)
    if bad:
        errs.append(f"numbers not present in the context: {bad[:5]} — remove or use only given numbers")
    if b in prior_bodies:
        errs.append("verbatim repeat of an earlier message")
    if mode == "accept":
        low = b.lower()
        if any(q in low for q in QUALIFYING):
            errs.append("merchant already said yes — do not ask qualifying questions (no 'would you'/'do you'/'how about'/'what if')")
    if re.search(r"i hope (you are|you're) doing well|reaching out today", b, re.I):
        errs.append("remove preamble")
    return errs


# --------------------------------------------------------------------------------------
# LLM prompts
# --------------------------------------------------------------------------------------
SYSTEM_COMPOSE = """You are Vera, magicpin's merchant-growth assistant on WhatsApp. You write ONE outbound WhatsApp message.

Hard rules:
- Use ONLY facts in the CONTEXT JSON. Never invent numbers, research, competitors, prices, dates or slots. Every number you write must appear in the context.
- Open with the salutation given, then go straight to WHY NOW (the trigger) in the first sentence. No preamble, no self-introduction, no "hope you are well".
- Anchor on 1-3 concrete, verifiable facts (numbers, dates, source citation, offer "Service @ ₹price"). Cite the source for research/compliance items (e.g. "— JIDA Oct 2026, p.14").
- Personalise with THIS merchant's own data (their numbers vs peer benchmarks, their active offers, signals, review themes, customer counts).
- Prefer service+price offers from the merchant's active offers or the category catalog over generic "% off".
- Add judgment: one smart, specific recommendation (can be contrarian if the data supports it).
- Exactly ONE call-to-action, in the last sentence, low friction (e.g. "Want me to draft it? Reply YES"). For customer booking flows, numbered slot choices are OK.
- No URLs. No taboo words. No ALL-CAPS hype. Max 1 emoji. 280-520 characters.
- Language: if hinglish=true, write natural Hindi-English code-mix in Roman script (mostly English with Hindi connectors like "aapke", "abhi", "kar doon?"); else English.
- Never mention internal terms like "trigger", "signal", "payload", "context", "suppression".
- If sending on behalf of the merchant to a customer: speak as the merchant's business ("<business> here"), warm, no medical claims, honour the customer's preferences and language.

Return STRICT JSON: {"body": "...", "cta": "open_ended|binary_yes_no|binary_confirm_cancel|multi_choice_slot|none", "rationale": "1-2 sentences: which facts you anchored on and why this lever", "key_fact": "the single most important fact used"}"""

SYSTEM_REPLY = """You are Vera, magicpin's merchant-growth assistant, continuing a WhatsApp conversation.

Rules:
- Use ONLY facts in the CONTEXT JSON; never invent numbers, names, prices, slots or research.
- Follow the INSTRUCTION exactly (it comes from a deterministic intent router).
- No re-introduction, no preamble, no URLs. 1 short paragraph (max ~450 chars). One clear next step at the end.
- Language: reply in the language style given (hinglish = natural Roman Hindi-English mix).
- Never repeat an earlier message verbatim.

Return STRICT JSON: {"body": "...", "cta": "open_ended|binary_yes_no|binary_confirm_cancel|multi_choice_slot|none", "rationale": "1 sentence"}"""


def _compose_user_prompt(facts: dict, strategy: tuple, send_as: str, hinglish: bool, errors: list[str] | None) -> str:
    goal, lever, cta, deliverable = strategy
    parts = [
        f"SEND_AS: {send_as}  (vera = to the merchant; merchant_on_behalf = to the merchant's customer)",
        f"SALUTATION: {facts['customer']['name'] if send_as == 'merchant_on_behalf' and facts.get('customer') else facts['merchant']['salutation']}",
        f"HINGLISH: {str(hinglish).lower()}",
        f"TONE: {TONE_HINT.get(facts.get('category', ''), 'peer, practical')}",
        f"GOAL FOR THIS TRIGGER ({facts['trigger']['kind']}): {goal}",
        f"LEVERS: {lever}",
        f"SUGGESTED CTA TYPE: {cta}; if they say yes you will deliver: {deliverable}",
        "CONTEXT JSON:",
        json.dumps(facts, ensure_ascii=False, default=str),
    ]
    if errors:
        parts.append("YOUR PREVIOUS DRAFT WAS REJECTED FOR: " + "; ".join(errors) + ". Fix these.")
    return "\n".join(parts)


# --------------------------------------------------------------------------------------
# Deterministic fallback composer
# --------------------------------------------------------------------------------------

def _pct(x: Any) -> str:
    try:
        return f"{abs(float(x)) * 100:.0f}%"
    except Exception:
        return str(x)


def fallback_compose(facts: dict, send_as: str, hinglish: bool) -> tuple[str, str]:
    kind = facts["trigger"]["kind"] or "update"
    p = facts["trigger"].get("payload") or {}
    m = facts["merchant"]
    sal = m["salutation"]
    perf = m.get("performance_30d") or {}
    item = facts["trigger"].get("resolved_digest_item")
    offer = (m.get("active_offers") or facts.get("category_catalog_offers") or [None])[0]
    ask_yes = "Kya main draft kar doon? Reply YES." if hinglish else "Want me to draft it? Reply YES."
    cta = STRATEGY.get(kind, DEFAULT_STRATEGY)[2]

    if send_as == "merchant_on_behalf" and facts.get("customer"):
        c = facts["customer"]
        name = c.get("name") or "there"
        biz = m.get("name")
        if kind == "wedding_package_followup":
            body = (f"Hi {name}, {biz} here. {p.get('days_to_wedding')} days to your wedding on {p.get('wedding_date')} — "
                    f"this is the right window to start the {humanize(p.get('next_step_window_open', 'prep program'))} after your trial on {p.get('trial_completed')}. "
                    + ("Slot block kar dein? Reply YES." if hinglish else "Shall we block your first session? Reply YES."))
            return body, "binary_yes_no"
        slots = p.get("available_slots") or p.get("next_session_options") or []
        if slots:
            labels = [s.get("label") for s in slots if isinstance(s, dict) and s.get("label")][:3]
            opts = ", ".join(f"{i + 1}) {l}" for i, l in enumerate(labels))
            last = _g(c, "relationship", "last_visit")
            if kind == "trial_followup":
                lead = f"Hi {name}, {biz} here. Hope the trial on {p.get('trial_date')} felt good!"
            else:
                due = humanize(p.get("service_due", "next visit"))
                lead = f"Hi {name}, {biz} here. Your {due} is due" + (f" (last visit {last})" if last else "") + "."
            mid = (f" Aapke liye slots ready hain: {opts}." if hinglish else f" Slots we've kept for you: {opts}.")
            tail = f" {offer} applies." if offer else ""
            end = " Reply with 1 or 2, ya apna time bata dijiye." if hinglish else " Reply with the number, or tell us a time that suits you."
            return lead + mid + tail + end, "multi_choice_slot"
        if kind == "chronic_refill_due":
            mols = ", ".join(p.get("molecule_list", [])[:4])
            runs = str(p.get("stock_runs_out_iso", ""))[:10]
            body = (f"Namaste {name}, {biz} here. Your monthly medicines ({mols}) are due to run out around {runs}. "
                    + ("Delivery aapke saved address pe kar dein? " if p.get("delivery_address_saved") and hinglish else
                       "Shall we deliver to your saved address? " if p.get("delivery_address_saved") else "")
                    + "Reply YES to confirm the refill.")
            return body, "binary_yes_no"
        body = (f"Hi {name}, {biz} here. It's been a while since your last visit"
                + (f" — {p.get('days_since_last_visit')} days" if p.get("days_since_last_visit") else "")
                + ". " + (f"{offer} is on right now. " if offer else "")
                + "Reply YES and we'll hold a slot for you this week.")
        return body, "binary_yes_no"

    if item:
        title = item.get("title", "")
        src = item.get("source", "")
        extra = ""
        if item.get("trial_n"):
            extra = f" ({item['trial_n']:,}-patient trial)" if isinstance(item.get("trial_n"), int) else ""
        action = item.get("actionable") or ""
        body = (f"{sal}, quick one from this week's {facts['category']} digest: {title}{extra}. "
                f"{action + '. ' if action else ''}"
                + (f"Aapke {PLACE.get(facts['category'], 'business')} ke liye 2-min summary + checklist bana doon? Reply YES." if hinglish else
                   "Want me to send a 2-min summary + checklist for your setup? Reply YES.")
                + (f" — {src}" if src else ""))
        return body, "binary_yes_no"

    if kind in ("perf_dip", "seasonal_perf_dip", "perf_spike"):
        metric = p.get("metric", "views")
        d = p.get("delta_pct")
        direction = "up" if (isinstance(d, (int, float)) and d > 0) else "down"
        base = f"{sal}, your {metric} are {direction} {_pct(d)} over the last {p.get('window', '7d')}"
        base += f" (30-day views {perf.get('views')}, calls {perf.get('calls')})." if perf.get("views") else "."
        if kind == "seasonal_perf_dip":
            base += " This is the usual seasonal lull, so no panic — best move now is keeping current members engaged."
        elif kind == "perf_spike" and p.get("likely_driver"):
            base += f" Likely driver: {humanize(p['likely_driver'])} — worth doubling down."
        elif offer:
            base += f" A fresh Google post featuring {offer} usually helps recover."
        return base + " " + ask_yes, "binary_yes_no"

    if kind == "renewal_due":
        body = (f"{sal}, your magicpin {p.get('plan', 'plan')} has {p.get('days_remaining')} days left"
                + (f" (renewal ₹{p.get('renewal_amount')})" if p.get("renewal_amount") else "")
                + f". Last 30 days: {perf.get('views')} views, {perf.get('calls')} calls from your listing. "
                + ("Renewal set up kar doon? Reply YES — koi auto-charge nahi." if hinglish else "Want me to set up the renewal? Reply YES — no auto-charge."))
        return body, "binary_yes_no"

    if kind == "competitor_opened":
        body = (f"{sal}, heads-up: {p.get('competitor_name', 'a new clinic')} opened {p.get('distance_km')} km away"
                + (f" with {p.get('their_offer')}" if p.get("their_offer") else "") + ". "
                + (f"Instead of a price war, let's lead with {offer} plus your reviews. " if offer else "Let's lead with your reviews, not price. ")
                + ask_yes)
        return body, "binary_yes_no"

    if kind == "festival_upcoming":
        body = (f"{sal}, {p.get('festival')} is on {p.get('date')}. "
                + (f"Good moment to push {offer} as a festive special. " if offer else "")
                + ask_yes)
        return body, "binary_yes_no"

    if kind == "ipl_match_today":
        wk = p.get("is_weeknight")
        body = (f"{sal}, {p.get('match')} tonight at {p.get('venue')}. "
                + ("Weeknight match — dine-in covers usually rise, so a match-night combo makes sense. " if wk else
                   "Weekend match — people mostly watch at home, so push delivery rather than dine-in. ")
                + (f"Your {offer} fits well. " if offer else "") + ask_yes)
        return body, "binary_yes_no"

    if kind == "review_theme_emerged":
        body = (f"{sal}, {p.get('occurrences_30d')} reviews in 30 days mention {humanize(p.get('theme'))}"
                + (f" — e.g. \"{p.get('common_quote')}\"" if p.get("common_quote") else "")
                + ". Replying politely + one visible fix stops it hurting your rating. "
                + ("Replies draft kar doon? Reply YES." if hinglish else "Want me to draft the replies? Reply YES."))
        return body, "binary_yes_no"

    if kind == "milestone_reached":
        body = (f"{sal}, you're at {p.get('value_now')} {humanize(p.get('metric', 'reviews')).replace('review count', 'reviews')} — just "
                f"{(p.get('milestone_value') or 0) - (p.get('value_now') or 0)} away from {p.get('milestone_value')}. "
                "A short thank-you note to recent happy customers usually closes that gap in a week. " + ask_yes)
        return body, "binary_yes_no"

    if kind == "gbp_unverified":
        body = (f"{sal}, your Google profile is still unverified — verified listings typically see ~{_pct(p.get('estimated_uplift_pct', 0.3))} more visibility. "
                f"Verification is via {humanize(p.get('verification_path', 'postcard or phone call'))}. "
                + ("Main process start kar doon? Reply YES." if hinglish else "Want me to start it for you? Reply YES."))
        return body, "binary_yes_no"

    if kind in ("winback_eligible", "dormant_with_vera"):
        body = (f"{sal}, " + (f"since your listing lapsed {p.get('days_since_expiry')} days ago, {p.get('lapsed_customers_added_since_expiry')} customers have lapsed. "
                              if p.get("days_since_expiry") else f"it's been {p.get('days_since_last_merchant_message', 'a few')} days since we spoke. ")
                + (f"Your 30-day views are at {perf.get('views')}. " if perf.get("views") else "")
                + ("Ek quick profile check karke 3 fixes bhej doon? Reply YES." if hinglish else "Shall I send you 3 quick fixes from a profile check? Reply YES."))
        return body, "binary_yes_no"

    if kind == "active_planning_intent":
        topic = humanize(p.get("intent_topic", "the plan"))
        body = (f"{sal}, here's a first cut for {topic}: " + (f"anchor it on {offer}, " if offer else "")
                + "run it as a fixed weekly slot, and announce it with one Google post + a WhatsApp broadcast. "
                + ("Confirm karein toh main set up kar doon? Reply CONFIRM." if hinglish else "Reply CONFIRM and I'll set it up."))
        return body, "binary_confirm_cancel"

    if kind == "curious_ask_due":
        body = (f"{sal}, quick question — which service is getting asked for the most this week? "
                "Tell me and I'll turn it into a Google post for you today.")
        return body, "open_ended"

    if kind == "category_seasonal":
        trends = ", ".join(re.sub(r"([+-]\d+)$", r"\1%", humanize(t)) for t in (p.get("trends") or [])[:4])
        body = f"{sal}, seasonal demand is shifting: {trends}. Worth re-arranging the front shelf this week. " + ask_yes
        return body, "binary_yes_no"

    # generic
    kv = "; ".join(f"{humanize(k)}: {v}" for k, v in list(p.items())[:3] if not isinstance(v, (dict, list)))
    body = f"{sal}, quick update — {humanize(kind)}" + (f" ({kv})" if kv else "") + ". " + (f"Your {offer} fits this well. " if offer else "") + ask_yes
    return body, cta


# --------------------------------------------------------------------------------------
# Public: compose an initial outbound
# --------------------------------------------------------------------------------------
async def compose(category: dict | None, merchant: dict, trigger: dict, customer: dict | None, prior_bodies: list[str]) -> dict:
    kind = trigger.get("kind") or "update"
    send_as = "merchant_on_behalf" if (trigger.get("scope") == "customer" and customer) else "vera"
    facts = build_facts(category, merchant, trigger, customer)
    if trigger.get("scope") == "customer" and not customer:
        # customer context missing: tell the merchant instead of guessing the customer's details
        facts["note"] = "Customer profile not available; write to the MERCHANT about this customer event and offer to send the reminder on their behalf."
    hinglish = wants_hinglish(merchant, customer if send_as == "merchant_on_behalf" else None)
    strategy = STRATEGY.get(kind, DEFAULT_STRATEGY)

    errors: list[str] | None = None
    source = "fallback"
    body, cta, rationale = None, strategy[2], None
    if llm.enabled():
        for attempt in range(2):
            out = await llm.complete_json(SYSTEM_COMPOSE, _compose_user_prompt(facts, strategy, send_as, hinglish, errors), timeout=8.5 if attempt == 0 else 6.0)
            if not out:
                break
            errs = validate_body(out.get("body"), facts, prior_bodies)
            if not errs:
                body = out["body"].strip()
                cta = out.get("cta") if out.get("cta") in CTA_VALUES else strategy[2]
                rationale = str(out.get("rationale") or "").strip()[:400]
                source = "llm"
                break
            errors = errs
    if body is None:
        body, cta = fallback_compose(facts, send_as, hinglish)
        if validate_body(body, facts, prior_bodies):
            body = body + " "  # never an exact repeat; numbers in fallback come straight from context
        rationale = None

    if not rationale:
        rationale = f"{kind} trigger (urgency {trigger.get('urgency')}); anchored on merchant's own data; lever: {strategy[1]}."
    rationale = f"{rationale} [composer={source}]"
    return {"body": body.strip(), "cta": cta, "send_as": send_as, "rationale": rationale, "facts": facts}


# --------------------------------------------------------------------------------------
# Reply handling
# --------------------------------------------------------------------------------------
AUTO_REPLY_PATTERNS = [
    r"thank(s| you) for (contacting|reaching|your message|messaging)",
    r"(will|shall) (get back|respond|revert|reply)( to you)? (shortly|soon|as soon)",
    r"our team will", r"automated (message|reply|assistant|response)", r"auto[- ]?reply",
    r"(currently|presently) (unavailable|away|closed)", r"outside (of )?(our )?(business|working) hours",
    r"we are (currently )?(closed|away)", r"i am an automated", r"main ek automated",
    r"jaankari ke liye.*shukriya", r"team tak pahuncha", r"hamari team",
    r"business hours", r"we have received your (message|query)",
]
OPT_OUT = [
    r"\bstop\b", r"unsubscribe", r"not interested", r"no interest", r"don'?t (message|contact|text|send)",
    r"do not (message|contact|text|send)", r"leave me alone", r"\bspam\b", r"useless", r"bothering",
    r"band karo", r"mat bhejo", r"nahi chahiye", r"nahin chahiye", r"interest nahi", r"block",
    r"fuck|bloody|idiot|stupid|bakwas|bekaar|chutiya|harami|shut up",
]
ACCEPT = [
    r"\byes\b", r"\byeah\b", r"\byep\b", r"\bsure\b", r"\bok(ay)?\b", r"go ahead", r"do it", r"let'?s do",
    r"let'?s go", r"\bproceed\b", r"\bconfirm", r"please do", r"sounds good", r"send (it|me|the)",
    r"\bhaa?n\b", r"\bhaanji\b", r"\bji\b", r"kar do", r"kardo", r"karo\b", r"theek hai", r"thik hai", r"\bchalo\b",
    r"bhej do", r"bhejo", r"\bdone\b", r"interested", r"i want", r"join", r"\bstart\b", r"what'?s next", r"whats next",
]
DEFER = [r"\blater\b", r"\bbusy\b", r"baad mein", r"baad me", r"\bkal\b", r"tomorrow", r"next week", r"in a meeting", r"call you back", r"abhi nahi"]
NEGATIVE_SOFT = [r"^\s*(no|nope|nahi|nahin|na)\b[\s.!]*$", r"no thanks", r"not now", r"not needed", r"no need"]
OFF_TOPIC = [r"\bgst\b", r"income tax", r"\bitr\b", r"\bloan\b", r"\bvisa\b", r"\bcricket score\b", r"\bpassport\b", r"electricity bill", r"\baccountant\b", r"\blegal notice\b"]


def _any(pats: list[str], text: str) -> bool:
    return any(re.search(p, text, re.I) for p in pats)


def classify(message: str, prior_inbound: list[str]) -> str:
    t = (message or "").strip()
    low = t.lower()
    if not t:
        return "empty"
    norm = re.sub(r"\W+", " ", low).strip()
    repeats = sum(1 for p in prior_inbound if re.sub(r"\W+", " ", p.lower()).strip() == norm)
    if _any(AUTO_REPLY_PATTERNS, low) or (repeats >= 1 and len(norm) > 25):
        return "auto_reply"
    if _any(OPT_OUT, low):
        return "opt_out"
    if _any(NEGATIVE_SOFT, low):
        return "decline"
    if _any(OFF_TOPIC, low):
        return "off_topic"
    if _any(DEFER, low) and not _any([r"go ahead", r"do it", r"kar do"], low):
        return "defer"
    if _any(ACCEPT, low):
        return "accept"
    if "?" in t or re.match(r"^(what|how|why|when|where|which|who|can|could|is|are|do|does|kya|kaise|kab|kitna)\b", low):
        return "question"
    return "other"


def detect_hinglish_turn(message: str) -> bool:
    if re.search(r"[ऀ-ॿ]", message):
        return True
    return bool(re.search(r"\b(haan|nahi|kar|karo|hai|hain|mujhe|aap|kya|kaise|bhej|theek|thik|chahiye|abhi|ji|kal|mein)\b", message.lower()))


INSTRUCTIONS = {
    "accept": ("The merchant has said YES / committed. Switch to ACTION MODE immediately: start with 'Done' or 'On it', say concretely what you are doing now "
               "({deliverable}), include a short ready-to-use draft/preview inline if possible, and end with a single CONFIRM-style step. "
               "Do NOT ask any qualifying question. Do NOT use the phrases 'would you', 'do you', 'how about', 'what if', 'can you tell'."),
    "question": "Answer the merchant's question directly and briefly using only the context facts (say honestly if the data isn't available), then give one next step tied to the original topic.",
    "off_topic": "The merchant asked something outside Vera's scope. Politely say that's best handled by their CA/relevant expert (one line, no advice), then bring it back to the original topic with one low-friction next step.",
    "other": "Acknowledge what the merchant said in one short phrase, respond usefully with one concrete fact from the context, and give one low-friction next step.",
}


def fallback_reply(intent: str, facts: dict | None, deliverable: str, hinglish: bool, merchant_msg: str, stage: int = 1) -> tuple[str, str]:
    sal = _g(facts, "merchant", "salutation", default="") if facts else ""
    name = f"{sal}, " if sal else ""
    if intent == "accept" and stage >= 2:
        if hinglish:
            return (f"Confirmed {sal} ✅ Yeh live ho raha hai. 48 ghante mein results (views/calls) yahin share karungi."), "none"
        return ("Confirmed ✅ Publishing it now. I'll share how it performs (views and calls) here in 48 hours."), "none"
    if intent == "accept":
        if hinglish:
            return (f"Done {sal}! Main abhi {deliverable} ready kar rahi hoon — next 10 min mein draft yahin bhej dungi. "
                    "Reply CONFIRM to publish it as-is.").replace("  ", " "), "binary_confirm_cancel"
        return (f"Done — on it. I'm preparing {deliverable} now and will send the draft here within 10 minutes. "
                "Reply CONFIRM and I'll publish it as soon as it's ready."), "binary_confirm_cancel"
    if intent == "off_topic":
        return (f"{name}that one's best handled by your CA — it's outside what I can help with. "
                f"Meanwhile, I can still get {deliverable} ready for you. Reply YES to proceed."), "binary_yes_no"
    if intent == "question":
        return (f"{name}good question — I'll check the exact details from your dashboard and share them here. "
                f"In parallel I can prepare {deliverable}. Reply YES to go ahead."), "binary_yes_no"
    return (f"{name}noted. Next step from my side: {deliverable}. Reply YES and I'll get it ready."), "binary_yes_no"


async def respond(conv: dict, facts: dict | None, message: str, intent: str, prior_bodies: list[str]) -> dict:
    """Compose a 'send' reply for accept/question/off_topic/other intents."""
    kind = _g(facts, "trigger", "kind", default=None) if facts else None
    deliverable = STRATEGY.get(kind, DEFAULT_STRATEGY)[3] if kind else "the draft"
    hinglish = detect_hinglish_turn(message) or (conv.get("hinglish", False))
    instruction = INSTRUCTIONS.get(intent, INSTRUCTIONS["other"]).format(deliverable=deliverable)
    if intent == "accept" and conv.get("accepts", 1) >= 2:
        instruction = ("The merchant has CONFIRMED. Say it is done/being published now, summarise in one line exactly what went live, "
                       "and say what you'll report back and when (no questions). CTA none.")
    body = cta = rationale = None
    source = "fallback"
    ctx = facts or {}
    if llm.enabled():
        errors = None
        history = conv.get("turns", [])[-6:]
        for attempt in range(2):
            user = "\n".join([
                f"INSTRUCTION: {instruction}",
                f"LANGUAGE: {'hinglish' if hinglish else 'english'}",
                f"SEND_AS: {conv.get('send_as', 'vera')}",
                "CONVERSATION SO FAR (oldest first):",
                json.dumps(history, ensure_ascii=False),
                f"LATEST MESSAGE FROM {conv.get('from_role', 'merchant').upper()}: {message}",
                "CONTEXT JSON:",
                json.dumps(ctx, ensure_ascii=False, default=str),
            ] + ([f"PREVIOUS DRAFT REJECTED FOR: {'; '.join(errors)}. Fix."] if errors else []))
            out = await llm.complete_json(SYSTEM_REPLY, user, timeout=8.5 if attempt == 0 else 5.5)
            if not out:
                break
            errs = validate_body(out.get("body"), ctx, prior_bodies, mode="accept" if intent == "accept" else "reply")
            if not errs:
                body = out["body"].strip()
                cta = out.get("cta") if out.get("cta") in CTA_VALUES else "open_ended"
                rationale = str(out.get("rationale") or "").strip()[:300]
                source = "llm"
                break
            errors = errs
    if body is None:
        body, cta = fallback_reply(intent, facts, deliverable, hinglish, message, conv.get("accepts", 1))
        if body in prior_bodies:
            body = body.replace("Reply", "Just reply", 1)
    rationale = f"intent={intent}; {rationale or 'deterministic reply for this intent'} [composer={source}]"
    return {"action": "send", "body": body, "cta": cta, "rationale": rationale}
