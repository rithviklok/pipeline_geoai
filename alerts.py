"""
alerts.py — Pluggable alert hooks (product-doc tickets 4.5-4.7).

Transport is intentionally undecided (open question in the plan): by default
every alert is logged loudly at ERROR level so it shows up in any log
aggregation already in place. Set the ALERT_WEBHOOK_URL environment variable
to also POST a JSON payload to that URL (e.g. a Slack incoming webhook, or an
internal notification service) — no code changes needed once a transport is
chosen.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

ALERT_WEBHOOK_ENV = "ALERT_WEBHOOK_URL"

# Fraction change in a headline number that triggers an "anomalous jump" alert.
ANOMALY_THRESHOLD = float(os.environ.get("ANOMALY_THRESHOLD_PCT", "25")) / 100.0

_ANOMALY_WATCHED_FIELDS = [
    "matched_count", "unmatched_count", "potential_defaulters", "total_mseva", "total_gis",
]


def _dispatch(kind: str, city: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
    payload = {"kind": kind, "city": city, "message": message, "context": context or {}}
    logger.error("[ALERT:%s] %s — %s", kind, city, message)

    webhook = os.environ.get(ALERT_WEBHOOK_ENV)
    if not webhook:
        return
    try:
        req = urllib.request.Request(
            webhook,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        logger.warning("Failed to deliver alert to %s: %s", ALERT_WEBHOOK_ENV, e)


def refresh_failed(city: str, error: str, context: Optional[Dict[str, Any]] = None) -> None:
    """Ticket 4.5: a refresh failed, or didn't start when scheduled."""
    _dispatch("REFRESH_FAILED", city, error, context)


def input_missing_or_stale(city: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
    """Ticket 4.6: an input source is missing or out of date."""
    _dispatch("INPUT_MISSING_OR_STALE", city, message, context)


def anomalous_jump(city: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
    """Ticket 4.7: a headline number jumped unexpectedly vs. last month."""
    _dispatch("ANOMALOUS_JUMP", city, message, context)


def check_anomalies(
    city: str,
    current_summary: Dict[str, Any],
    previous_summary: Optional[Dict[str, Any]],
) -> None:
    """Compare this run's summary against the last published run and alert
    on any watched headline number that moved by more than ANOMALY_THRESHOLD.
    No-op if there is no previous summary to compare against (first run)."""
    if not previous_summary:
        return
    for field in _ANOMALY_WATCHED_FIELDS:
        prev = previous_summary.get(field)
        curr = current_summary.get(field)
        if not isinstance(prev, (int, float)) or not isinstance(curr, (int, float)) or prev == 0:
            continue
        change = abs(curr - prev) / abs(prev)
        if change > ANOMALY_THRESHOLD:
            anomalous_jump(
                city,
                f"'{field}' moved {change:.0%} vs. last published run "
                f"({prev:,} \u2192 {curr:,}), exceeding the {ANOMALY_THRESHOLD:.0%} threshold.",
                context={"field": field, "previous": prev, "current": curr},
            )
