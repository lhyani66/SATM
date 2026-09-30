"""Sending a password-reset code by email.

Deliberately stdlib smtplib rather than a provider's SDK: every transactional
mail service (Brevo, Mailgun, Mailjet, Postmark, Gmail, ...) offers an SMTP
relay, so which one SATM uses is an environment decision and never a code
change.

Configure in the environment (on Render: Dashboard -> Environment):

    SMTP_HOST      the relay's hostname, e.g. smtp-relay.brevo.com
    SMTP_PORT      optional, defaults to 587 (STARTTLS); 465 switches to SSL
    SMTP_USER      the relay's login
    SMTP_PASSWORD  the relay's key, or an app password for Gmail
    MAIL_FROM      the verified sender, e.g. "SATM <no-reply@example.com>"

With nothing configured is_configured() is False and the caller decides what to
do about it, rather than this module guessing: local development prints the code
to the log so the whole flow can be tested without an account anywhere, and
production refuses the request instead of silently dropping mail.
"""

import os
import smtplib
import ssl
from email.message import EmailMessage

SMTP_HOST     = os.environ.get('SMTP_HOST', '')
SMTP_PORT     = int(os.environ.get('SMTP_PORT') or 587)
SMTP_USER     = os.environ.get('SMTP_USER', '')
SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD', '')
# Most relays reject a sender they have not verified, so this has to be set
# explicitly — guessing from SMTP_USER would bounce on half of them.
MAIL_FROM     = os.environ.get('MAIL_FROM', '')


def is_configured():
    """True when there is somewhere to send mail and an address to send it from."""
    return bool(SMTP_HOST and MAIL_FROM)


def send_reset_code(to_email, code, minutes):
    """Send one reset code. Raises on failure — the caller reports it."""
    msg = EmailMessage()
    # The code leads the subject so it is readable from a notification without
    # opening the message, which is the whole point of a short numeric code.
    msg['Subject'] = f'{code} is your SATM password reset code'
    msg['From']    = MAIL_FROM
    msg['To']      = to_email
    msg.set_content(
        f'Your SATM password reset code is {code}\n\n'
        f'Type it into the app to choose a new password. The code stops working '
        f'in {minutes} minutes.\n\n'
        'If you did not ask to reset your password, you can ignore this email — '
        'nothing has changed on your account.\n'
    )

    if SMTP_PORT == 465:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context(),
                              timeout=20) as smtp:
            if SMTP_USER:
                smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.send_message(msg)
        return

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as smtp:
        smtp.starttls(context=ssl.create_default_context())
        if SMTP_USER:
            smtp.login(SMTP_USER, SMTP_PASSWORD)
        smtp.send_message(msg)
