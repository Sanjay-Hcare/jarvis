import imaplib
from datetime import datetime, timezone

import pytest

from jarvis.gcalendar import get_calendar
from jarvis.mail import EmailSummary, check_email, unread_emails
from jarvis.tasks import default_tools
from jarvis.tools import ToolBox


def secrets_of(**values):
    return lambda service, name: values.get(name)


def stored_password(service, name):
    return "pw"


def test_email_summary_lists_newest_first():
    def fetch(address, password, host, days):
        assert (address, password, host, days) == ("me@example.com", "pw", "imap.example.com", 3)
        return 7, [EmailSummary("Priya", "Lunch on Friday?"), EmailSummary("Bank", "Your statement")]

    report = check_email("me@example.com", "imap.example.com", "svc", days=3, secret=stored_password, fetch=fetch)

    assert report == (
        "7 unread emails from the last 3 days. The newest 2: "
        "1) from Priya: Lunch on Friday?; 2) from Bank: Your statement."
    )


def test_email_edge_cases():
    def none(*args):
        return 0, []

    def one(*args):
        return 1, [EmailSummary("A", "B")]

    def broken(*args):
        raise imaplib.IMAP4.error("login failed")

    assert check_email("", secret=stored_password, fetch=none).startswith("Error: email is not set up")
    assert check_email("me@x.com", secret=secrets_of(), fetch=none).startswith("Error: no email password")
    assert check_email("me@x.com", secret=stored_password, fetch=none) == "There are no unread emails from the last 2 days."
    assert check_email("me@x.com", secret=stored_password, fetch=broken) == (
        "Error: could not check the mailbox (login failed)."
    )
    assert check_email("me@x.com", days=1, secret=stored_password, fetch=one) == "1 unread email from the last day. 1) from A: B."


class FakeIMAP:
    """Just enough of an IMAP server to check what is asked of it."""

    def __init__(self, host):
        self.host = host
        self.commands = []
        self.searches = []
        self.messages = {
            b"1": b"From: Old <old@x.com>\r\nSubject: first\r\n\r\n",
            b"2": b"From: =?utf-8?b?UHJpeWEgUmFv?= <p@x.com>\r\nSubject: =?utf-8?q?Caf=C3=A9_tomorrow?=\r\n\r\n",
            b"3": b"From: bare@x.com\r\n\r\n",
        }

    def login(self, address, password):
        self.commands.append(("login", address))

    def select(self, mailbox, readonly=False):
        self.commands.append(("select", mailbox, readonly))

    def uid(self, command, *args):
        self.commands.append((command, *args))
        if command == "search":
            self.searches.append(args[1:])
            if args[1] == "X-GM-RAW":
                return "BAD", [b""]
            return "OK", [b"1 2 3"]
        return "OK", [(b"header", self.messages[args[0]])]

    def logout(self):
        self.commands.append(("logout",))


def test_mailbox_is_opened_read_only_and_headers_decoded():
    servers = []

    def connect(host):
        servers.append(FakeIMAP(host))
        return servers[-1]

    total, emails = unread_emails("me@gmail.com", "pw", "imap.gmail.com", days=4, limit=2, connect=connect)

    assert total == 3
    assert emails == [EmailSummary("bare@x.com", "(no subject)"), EmailSummary("Priya Rao", "Café tomorrow")]
    commands = servers[0].commands
    assert ("select", "INBOX", True) in commands
    assert commands[-1] == ("logout",)
    # Only header fields are requested, with PEEK so nothing is marked as read.
    assert all("PEEK" in c[2] and "HEADER.FIELDS" in c[2] for c in commands if c[0] == "fetch")
    # Gmail's search was refused by this server, so the standard one was used.
    gmail_search, standard_search = servers[0].searches
    assert gmail_search == ("X-GM-RAW", '"in:inbox is:unread category:primary newer_than:4d"')
    assert standard_search[:2] == ("UNSEEN", "SINCE")


def at(hour, minute=0):
    local = datetime(2026, 3, 5, hour, minute).astimezone()
    return local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def fixed_now():
    return datetime(2026, 3, 5, 10, 0)


CREDENTIALS = secrets_of(calendar_url="https://script.google.com/x", calendar_secret="key")


def test_calendar_today():
    sent = {}

    def post(url, payload):
        sent.update(payload, url=url)
        return {
            "ok": True,
            "events": [
                {"title": "Team call", "all_day": False, "date": "2026-03-05", "start": at(15, 30)},
                {"title": "Gym", "all_day": False, "date": "2026-03-05", "start": at(7)},
                {"title": "Holiday", "all_day": True, "date": "2026-03-05", "start": at(0)},
            ],
        }

    report = get_calendar("today", secret=CREDENTIALS, post=post, now=fixed_now)

    assert report == "3 events today. all day: Holiday; 7:00 AM: Gym; 3:30 PM: Team call."
    assert sent["url"] == "https://script.google.com/x"
    assert sent["secret"] == "key" and sent["action"] == "list"
    assert sent["start"].startswith("2026-03-05T00:00:00") and sent["end"].startswith("2026-03-06T00:00:00")


def test_calendar_ranges():
    seen = []

    def post(url, payload):
        seen.append((payload["start"][:10], payload["end"][:10]))
        return {"ok": True, "events": []}

    assert get_calendar("tomorrow", secret=CREDENTIALS, post=post, now=fixed_now) == "Nothing is on the calendar tomorrow."
    assert (
        get_calendar("this week", secret=CREDENTIALS, post=post, now=fixed_now)
        == "Nothing is on the calendar in the next seven days."
    )
    assert get_calendar("gibberish", secret=CREDENTIALS, post=post, now=fixed_now) == "Nothing is on the calendar today."
    assert seen == [("2026-03-06", "2026-03-07"), ("2026-03-05", "2026-03-12"), ("2026-03-05", "2026-03-06")]


def test_calendar_week_names_the_day():
    def post(url, payload):
        return {"ok": True, "events": [{"title": "Exam", "all_day": True, "date": "2026-03-09", "start": at(0)}]}

    report = get_calendar("week", secret=CREDENTIALS, post=post, now=fixed_now)

    assert report == "1 event in the next seven days. Monday all day: Exam."


def test_calendar_problems_are_reported():
    def unused(url, payload):
        raise AssertionError("should not be called")

    def refused(url, payload):
        return {"ok": False, "error": "unauthorized"}

    def down(url, payload):
        raise RuntimeError("the calendar could not be reached (timeout)")

    assert get_calendar("today", secret=secrets_of(), post=unused).startswith("Error: the calendar is not connected")
    assert (
        get_calendar("today", secret=CREDENTIALS, post=refused, now=fixed_now)
        == "Error: the calendar refused the request (unauthorized)."
    )
    assert (
        get_calendar("today", secret=CREDENTIALS, post=down, now=fixed_now)
        == "Error: the calendar could not be reached (timeout)."
    )


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Any new mail?", ["check_email"]),
        ("Do I have unread emails?", ["check_email"]),
        ("check my inbox", ["check_email"]),
        ("What's on my calendar today?", ["get_calendar"]),
        ("Do I have any meetings tomorrow?", ["get_calendar"]),
        ("What's on today?", ["get_calendar"]),
        ("Am I free this week?", ["get_calendar"]),
    ],
)
def test_email_and_calendar_requests_offer_the_right_tool(message, expected):
    assert [tool.name for tool in ToolBox(default_tools()).relevant(message)] == expected
