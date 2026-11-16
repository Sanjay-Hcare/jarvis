"""Store the email password and calendar link in the system credential store.

    python -m jarvis.setup_accounts

Nothing is written to a file in this folder; secrets go to Windows Credential
Manager (or the macOS Keychain / Linux Secret Service).
"""

from __future__ import annotations

import getpass
import secrets
import sys

from jarvis.config import load_settings
from jarvis.gcalendar import SECRET_KEY, URL_KEY, get_calendar
from jarvis.mail import check_email


def main() -> int:
    try:
        import keyring
    except ImportError:
        print('This needs the keyring package: pip install -e ".[accounts]"', file=sys.stderr)
        return 2
    settings = load_settings()
    service = settings.keyring_service

    print("Email")
    if not settings.email_address:
        print('  Add  email_address = "you@example.com"  to config.toml first, then run this again.')
    else:
        print(f"  Account: {settings.email_address} on {settings.imap_host}")
        print("  For Gmail, create an app password at https://myaccount.google.com/apppasswords")
        password = getpass.getpass("  App password (leave empty to keep the current one): ").replace(" ", "")
        if password:
            keyring.set_password(service, settings.email_address, password)
        print("  " + check_email(settings.email_address, settings.imap_host, service).split(". Newest")[0])

    print("\nCalendar")
    key = keyring.get_password(service, SECRET_KEY)
    if not key:
        key = secrets.token_urlsafe(32)
        keyring.set_password(service, SECRET_KEY, key)
    print("  Follow the steps at the top of docs/calendar_script.gs, using this key as SECRET:")
    print(f"    {key}")
    url = input("  Web app URL (leave empty to keep the current one): ").strip()
    if url:
        if not url.startswith("https://script.google.com/"):
            print("  That does not look like an Apps Script web app URL; not saved.")
        else:
            keyring.set_password(service, URL_KEY, url)
    print("  " + get_calendar("today", service))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
