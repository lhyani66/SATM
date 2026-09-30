"""
Proves the password-reset flow, without a server or a mail account.

    python reset_test.py

Runs Flask's test client against a throwaway SQLite file and stands in for the
mail relay, so the code it asserts on is the code the user would have been sent.
Standard library only.
"""
import os
import tempfile
from datetime import timedelta
from pathlib import Path

_db = os.path.join(tempfile.mkdtemp(), 'reset_test.db')
os.environ['DATABASE_URL'] = 'sqlite:///' + Path(_db).as_posix()

import app as satm  # noqa: E402  — must follow the DATABASE_URL above

# Stand in for the relay: the code that would have been emailed lands here.
mail = {}
satm.mailer.is_configured = lambda: True
satm.mailer.send_reset_code = lambda to, code, minutes: mail.update(to=to, code=code)

client = satm.app.test_client()
EMAIL, OLD, NEW = 'reset@test.local', 'Before!2026x', 'After!2026x'


def post(path, body):
    r = client.post(path, json=body)
    return r.status_code, r.get_json()


def fresh_code():
    """Ask for a code, stepping past the one-a-minute throttle."""
    mail.clear()
    with satm.app.app_context():
        user = satm.User.query.filter_by(email=EMAIL).first()
        for row in satm.PasswordReset.query.filter_by(user_id=user.id):
            row.created_at -= timedelta(seconds=satm.RESET_RESEND_SECONDS + 1)
        satm.db.session.commit()
    s, r = post('/api/forgot-password', {'email': EMAIL})
    assert s == 200 and mail.get('code'), (s, r, mail)
    return mail['code']


s, r = post('/api/register', {'email': EMAIL, 'password': OLD})
assert s == 201, (s, r)
client.post('/api/logout')
print('register        ok')

# An address with no account must be answered exactly like one that has an
# account, or this endpoint tells you who is registered.
s, unknown = post('/api/forgot-password', {'email': 'nobody@test.local'})
assert s == 200 and not mail, (s, unknown, mail)
print('unknown email   ok  (same answer, no mail sent)')

s, known = post('/api/forgot-password', {'email': EMAIL})
assert s == 200 and known == unknown, (known, unknown)
code = mail['code']
assert mail['to'] == EMAIL and len(code) == 6 and code.isdigit(), mail
print(f'code sent       ok  ({code})')

# Asking again straight away is throttled, and must not look any different.
mail.clear()
s, r = post('/api/forgot-password', {'email': EMAIL})
assert s == 200 and r == known and not mail, (s, r, mail)
print('resend throttle ok  (no second code within the minute)')

s, r = post('/api/reset-password', {'email': EMAIL, 'code': '000000', 'password': NEW})
assert s == 400, (s, r)
print('wrong code      ok  ', r['error'])

s, r = post('/api/reset-password', {'email': EMAIL, 'code': code, 'password': NEW})
assert s == 200, (s, r)
print('reset           ok  ', r['message'])

s, r = post('/api/login', {'email': EMAIL, 'password': OLD})
assert s == 401, ('old password still works', s, r)
s, r = post('/api/login', {'email': EMAIL, 'password': NEW})
assert s == 200, (s, r)
client.post('/api/logout')
print('new password    ok  (and the old one is dead)')

# A code is single use: replaying it must not work a second time.
s, r = post('/api/reset-password', {'email': EMAIL, 'code': code, 'password': 'Third!2026x'})
assert s == 400, ('code was reusable', s, r)
print('code reuse      ok  ', r['error'])

# An expired code is refused even though it is the right code.
code = fresh_code()
with satm.app.app_context():
    row = satm.PasswordReset.query.order_by(satm.PasswordReset.id.desc()).first()
    row.expires_at -= timedelta(minutes=satm.RESET_CODE_TTL_MIN + 1)
    satm.db.session.commit()
s, r = post('/api/reset-password', {'email': EMAIL, 'code': code, 'password': 'Fourth!2026x'})
assert s == 400, ('expired code accepted', s, r)
print('expired code    ok  ', r['error'])

# Guessing is capped, so six digits cannot simply be enumerated.
fresh_code()
for n in range(satm.RESET_MAX_ATTEMPTS):
    s, r = post('/api/reset-password', {'email': EMAIL, 'code': '000001', 'password': 'Fifth!2026x'})
    assert s == 400, (n, s, r)
s, r = post('/api/reset-password', {'email': EMAIL, 'code': '000001', 'password': 'Fifth!2026x'})
assert s == 429, ('no attempt limit', s, r)
print('attempt limit   ok  ', r['error'])

s, r = post('/api/forgot-password', {'email': ''})
assert s == 400, (s, r)
s, r = post('/api/reset-password', {'email': EMAIL, 'code': '', 'password': ''})
assert s == 400, (s, r)
print('missing fields  ok')

print('\nALL GOOD — password reset works end to end.')
