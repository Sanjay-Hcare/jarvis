"""Read-only look at the inbox over IMAP. Nothing is sent, moved or marked as read."""

from __future__ import annotations

import email
import email.header
import email.utils
import imaplib
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable

HEADER_QUERY = "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])"


@dataclass(frozen=True)
class EmailSummary:
    sender: str
    subject: str


def get_secret(service: str, name: str) -> str | None:
    """Read a secret from the operating system's credential store."""
    import keyring

    return keyring.get_password(service, name)


def _decode(value: str | None) -> str:
    if not value:
        return ""
    parts = []
    for text, charset in email.header.decode_header(value):
        if isinstance(text, bytes):
            try:
                text = text.decode(charset or "utf-8", "replace")
            except LookupError:
                text = text.decode("utf-8", "replace")
        parts.append(text)
    return " ".join("".join(parts).split())


def _sender_name(raw: str | None) -> str:
    name, address = email.utils.parseaddr(_decode(raw))
    return name or address or "unknown sender"


def _unread_ids(imap: Any, host: str, days: int) -> list[bytes]:
    """Unread mail from the last few days.

    A long-neglected inbox can hold thousands of unread messages; "you have
    6,825 unread emails" answers nothing, so only recent ones are counted.
    """
    if "gmail" in host:
        # Gmail's inbox also holds promotions and social mail; only the
        # Primary tab is what people mean by "new mail".
        query = f'"in:inbox is:unread category:primary newer_than:{days}d"'
        status, data = imap.uid("search", None, "X-GM-RAW", query)
        if status == "OK":
            return data[0].split()
    since = (date.today() - timedelta(days=days)).strftime("%d-%b-%Y")
    status, data = imap.uid("search", None, "UNSEEN", "SINCE", since)
    return data[0].split() if status == "OK" else []


def unread_emails(
    address: str,
    password: str,
    host: str = "imap.gmail.com",
    days: int = 2,
    limit: int = 5,
    connect: Callable[[str], Any] = imaplib.IMAP4_SSL,
) -> tuple[int, list[EmailSummary]]:
    """Return how many unread emails there are and the newest few, newest first."""
    imap = connect(host)
    try:
        imap.login(address, password)
        # readonly: fetching must never mark anything as read.
        imap.select("INBOX", readonly=True)
        ids = _unread_ids(imap, host, days)
        summaries = []
        for uid in reversed(ids[-limit:]):
            _status, data = imap.uid("fetch", uid, HEADER_QUERY)
            if not data or not isinstance(data[0], tuple):
                continue
            message = email.message_from_bytes(data[0][1])
            summaries.append(
                EmailSummary(_sender_name(message["From"]), _decode(message["Subject"]) or "(no subject)")
            )
        return len(ids), summaries
    finally:
        try:
            imap.logout()
        except Exception:
            pass


def check_email(
    address: str = "",
    host: str = "imap.gmail.com",
    keyring_service: str = "jarvis",
    days: int = 2,
    secret: Callable[[str, str], str | None] = get_secret,
    fetch: Callable[..., tuple[int, list[EmailSummary]]] = unread_emails,
) -> str:
    """Describe the recent unread mail in words the assistant can relay."""
    period = "day" if days == 1 else f"{days} days"
    if not address:
        return "Error: email is not set up. Add email_address to config.toml."
    try:
        password = secret(keyring_service, address)
    except ImportError:
        return 'Error: email needs the keyring package (pip install -e ".[accounts]").'
    if not password:
        return "Error: no email password is stored. Run: python -m jarvis.setup_accounts"
    try:
        total, emails = fetch(address, password, host, days)
    except (imaplib.IMAP4.error, OSError) as exc:
        return f"Error: could not check the mailbox ({exc})."
    if total == 0:
        return f"There are no unread emails from the last {period}."
    listed = "; ".join(f"{i}) from {mail.sender}: {mail.subject[:90]}" for i, mail in enumerate(emails, 1))
    plural = "email" if total == 1 else "emails"
    shown = "" if total <= len(emails) else f" The newest {len(emails)}:"
    return f"{total} unread {plural} from the last {period}.{shown} {listed}."
