"""The built-in tools: time, weather, opening apps, screenshots, locking the screen."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from jarvis.gcalendar import get_calendar
from jarvis.mail import check_email
from jarvis.tools import Tool

# --- time -------------------------------------------------------------------


def get_time(now: Callable[[], datetime] = datetime.now) -> str:
    current = now()
    return current.strftime("It is %I:%M %p on %A, %d %B %Y.").replace(" 0", " ")


# --- weather ----------------------------------------------------------------

# WMO weather interpretation codes used by Open-Meteo.
_WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "violent showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


# Older city names that the geocoder only knows under their current name
# (searching "Bangalore" otherwise finds a neighbourhood in Karachi).
_CITY_ALIASES = {
    "bangalore": "Bengaluru",
    "bombay": "Mumbai",
    "madras": "Chennai",
    "calcutta": "Kolkata",
    "poona": "Pune",
    "mysore": "Mysuru",
    "mangalore": "Mangaluru",
    "trivandrum": "Thiruvananthapuram",
    "cochin": "Kochi",
    "gurgaon": "Gurugram",
}


def _get_json(url: str, attempts: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=8) as response:
                return json.load(response)
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"the weather service could not be reached ({last_error})")


def get_weather(city: str = "", home_city: str = "", fetch: Callable[[str], dict[str, Any]] = _get_json) -> str:
    """Current conditions and today's forecast from Open-Meteo (free, no account)."""
    city = (city or home_city).strip()
    if not city:
        return "Error: no city was given. Ask which city."
    city = _CITY_ALIASES.get(city.lower(), city)
    query = urllib.parse.urlencode({"name": city, "count": 5, "language": "en"})
    places = fetch(f"https://geocoding-api.open-meteo.com/v1/search?{query}").get("results") or []
    if not places:
        return f"Error: no place called {city} was found."
    # Several places can share a name; the most populous is almost always the one meant.
    place = max(places, key=lambda item: item.get("population") or 0)
    query = urllib.parse.urlencode(
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": 1,
            "timezone": "auto",
        }
    )
    data = fetch(f"https://api.open-meteo.com/v1/forecast?{query}")
    now, today = data["current"], data["daily"]
    sky = _WEATHER_CODES.get(now.get("weather_code"), "unknown conditions")
    where = ", ".join(part for part in (place.get("name"), place.get("country")) if part)
    return (
        f"{where}: {sky}, {round(now['temperature_2m'])}°C (feels like {round(now['apparent_temperature'])}°C), "
        f"wind {round(now['wind_speed_10m'])} km/h. Today: high {round(today['temperature_2m_max'][0])}°C, "
        f"low {round(today['temperature_2m_min'][0])}°C, {today['precipitation_probability_max'][0]}% chance of rain."
    )


# --- apps -------------------------------------------------------------------

# Only these can be opened. Each entry is a fixed command or address, so
# nothing the model says is ever passed to a shell.
APPS: dict[str, list[str] | str] = {
    "notepad": ["notepad.exe"],
    "calculator": ["calc.exe"],
    "file explorer": ["explorer.exe"],
    "paint": ["mspaint.exe"],
    "task manager": ["taskmgr.exe"],
    "command prompt": ["cmd.exe"],
    "settings": "ms-settings:",
    "browser": "https://www.google.com",
}


def open_app(app: str = "", launch: Callable[[list[str] | str], None] | None = None) -> str:
    name = app.strip().lower()
    target = APPS.get(name)
    if target is None:
        return f"Error: I can only open {', '.join(sorted(APPS))}."
    if os.name != "nt" and name != "browser":
        return "Error: opening apps is only supported on Windows."
    (launch or _launch)(target)
    return f"Opened {name}."


def _launch(target: list[str] | str) -> None:
    if isinstance(target, list):
        # CREATE_NEW_CONSOLE gives console apps such as cmd their own window.
        flags = subprocess.CREATE_NEW_CONSOLE if target[0] == "cmd.exe" else 0
        subprocess.Popen(target, creationflags=flags)
    elif target.startswith("http"):
        webbrowser.open_new_tab(target)
    else:
        os.startfile(target)  # noqa: S606 - fixed ms-settings: address from APPS


# --- screenshot -------------------------------------------------------------


def take_screenshot(folder: str = "", grab: Callable[[], Any] | None = None, now: Callable[[], datetime] = datetime.now) -> str:
    if grab is None:
        try:
            from PIL import ImageGrab
        except ImportError:
            return "Error: screenshots need the pillow package (pip install pillow)."
        grab = lambda: ImageGrab.grab(all_screens=True)  # noqa: E731
    target_dir = Path(folder).expanduser() if folder else Path.home() / "Pictures" / "Jarvis"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"screenshot-{now():%Y%m%d-%H%M%S}.png"
    grab().save(path)
    return f"Saved a screenshot to {path}."


# --- lock -------------------------------------------------------------------


def lock_computer(lock: Callable[[], Any] | None = None) -> str:
    if lock is None:
        if os.name != "nt":
            return "Error: locking is only supported on Windows."
        import ctypes

        lock = ctypes.windll.user32.LockWorkStation
    lock()
    return "The computer is locked."


# --- registry ---------------------------------------------------------------


def default_tools(
    home_city: str = "",
    screenshot_folder: str = "",
    email_address: str = "",
    imap_host: str = "imap.gmail.com",
    keyring_service: str = "jarvis",
    email_recent_days: int = 2,
) -> list[Tool]:
    return [
        Tool(
            name="check_email",
            description="Check the user's inbox for recent unread emails: how many, who from, and the subjects.",
            run=lambda: check_email(email_address, imap_host, keyring_service, email_recent_days),
            triggers=r"\b(e-?mails?|mails?|inbox)\b",
        ),
        Tool(
            name="get_calendar",
            description="Look up the events on the user's calendar for today, tomorrow or the coming week.",
            run=lambda day="today": get_calendar(day, keyring_service),
            parameters={
                "type": "object",
                "properties": {"day": {"type": "string", "enum": ["today", "tomorrow", "week"]}},
                "required": ["day"],
            },
            triggers=r"\b(calendar|schedule|agenda|meetings?|appointments?|events?|"
            r"what(?:'s| is| do i have)? on (?:today|tomorrow|this week|my)|am i (?:free|busy)|"
            r"(?:any|my) plans)\b",
        ),
        Tool(
            name="get_time",
            description="Get the current local time and date.",
            run=get_time,
            triggers=r"\b(what(?:'s| is)? (?:the )?(?:time|date|day)|(?:the|current) (?:time|date)\b|"
            r"what day|which day|today'?s date|date today|time is it|time now)",
        ),
        Tool(
            name="get_weather",
            description="Get the current weather and today's forecast for a city.",
            run=lambda city="": get_weather(city, home_city=home_city),
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name, for example Mumbai"}},
                "required": [] if home_city else ["city"],
            },
            triggers=r"\b(weather|forecast|temperature|rain(?:ing|y)?|umbrella|sunny|humid|"
            r"how (?:hot|cold|warm)|(?:hot|cold|warm) (?:outside|today|is it))\b",
        ),
        Tool(
            name="open_app",
            description="Open an application on the user's computer.",
            confirmation="Done, it's open.",
            run=open_app,
            parameters={
                "type": "object",
                "properties": {"app": {"type": "string", "enum": sorted(APPS)}},
                "required": ["app"],
            },
            triggers=r"\b(open|launch|start|fire up|bring up|pull up)\b",
        ),
        Tool(
            name="take_screenshot",
            description="Take a screenshot of the user's screen and save it as an image file.",
            confirmation="Screenshot saved.",
            run=lambda: take_screenshot(screenshot_folder),
            triggers=r"\b(screen ?shot|capture (?:my |the )?screen|screen capture|snip)\b",
        ),
        Tool(
            name="lock_computer",
            description="Lock the user's computer screen.",
            confirmation="Locked.",
            run=lock_computer,
            triggers=r"\block (?:my |the |this )?(?:computer|pc|laptop|screen|system|machine)\b|\block it\b",
        ),
    ]
