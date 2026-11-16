"""Read Google Calendar through a small Apps Script web app in the user's own account.

The script (see docs/calendar_script.gs) runs as the calendar's owner and is
protected by a secret key, so no Google Cloud project or OAuth consent screen
is needed. Its URL and key are kept in the operating system's credential store.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Callable

from jarvis.mail import get_secret

URL_KEY = "calendar_url"
SECRET_KEY = "calendar_secret"

# Spoken range -> (first day offset, number of days)
RANGES = {"today": (0, 1), "tomorrow": (1, 1), "week": (0, 7)}


def _post(url: str, payload: dict[str, Any], attempts: int = 3) -> dict[str, Any]:
    """POST JSON. Google answers with a redirect that urllib follows as a GET,
    which is how Apps Script hands back the result."""
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    last_error: Exception | None = None
    for _attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
    raise RuntimeError(f"the calendar could not be reached ({last_error})")


def _when(event: dict[str, Any]) -> str:
    if event.get("all_day"):
        return "all day"
    start = datetime.fromisoformat(str(event["start"]).replace("Z", "+00:00")).astimezone()
    return start.strftime("%I:%M %p").lstrip("0")


def get_calendar(
    day: str = "today",
    keyring_service: str = "jarvis",
    secret: Callable[[str, str], str | None] = get_secret,
    post: Callable[[str, dict[str, Any]], dict[str, Any]] = _post,
    now: Callable[[], datetime] = datetime.now,
) -> str:
    """Describe the events for today, tomorrow or the next seven days."""
    day = (day or "today").strip().lower()
    if day not in RANGES:
        day = "week" if "week" in day else "tomorrow" if "tomorrow" in day else "today"
    try:
        url, key = secret(keyring_service, URL_KEY), secret(keyring_service, SECRET_KEY)
    except ImportError:
        return 'Error: the calendar needs the keyring package (pip install -e ".[accounts]").'
    if not (url and key):
        return "Error: the calendar is not connected. Run: python -m jarvis.setup_accounts"

    offset, days = RANGES[day]
    start = now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=offset)
    end = start + timedelta(days=days)
    try:
        result = post(url, {"secret": key, "action": "list", "start": start.isoformat(), "end": end.isoformat()})
    except RuntimeError as exc:
        return f"Error: {exc}."
    if not result.get("ok"):
        return f"Error: the calendar refused the request ({result.get('error')})."

    events = sorted(result.get("events", []), key=lambda event: str(event.get("start", "")))
    label = {"today": "today", "tomorrow": "tomorrow", "week": "in the next seven days"}[day]
    if not events:
        return f"Nothing is on the calendar {label}."
    parts = []
    for event in events:
        prefix = ""
        if day == "week":
            prefix = datetime.strptime(event["date"], "%Y-%m-%d").strftime("%A") + " "
        parts.append(f"{prefix}{_when(event)}: {str(event.get('title', 'untitled'))[:80]}")
    plural = "event" if len(events) == 1 else "events"
    return f"{len(events)} {plural} {label}. " + "; ".join(parts) + "."
