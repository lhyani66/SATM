"""
Proves the API works end to end: register → predict → save → tick → list.

    python smoke_test.py                       # against local server
    python smoke_test.py https://satm.onrender.com

Standard library only. Creates a throwaway account each run.
"""
import json, sys, time, urllib.request
from http.cookiejar import CookieJar

BASE = (sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:5000').rstrip('/')
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))

def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={'Content-Type': 'application/json'})
    try:
        with opener.open(req, timeout=90) as r:          # 90s: Render cold start
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)

email = f'smoke-{int(time.time())}@test.local'

s, r = call('POST', '/api/register', {'email': email, 'password': 'Smoke!2026x'})
assert s == 201 and r['email'] == email, (s, r)
print('register  ok ', r)

s, r = call('GET', '/api/me')
assert s == 200, (s, r)
print('me        ok ', r)

s, r = call('POST', '/api/predict', {'text': 'finish bio hw ASAP exam tomorrow', 'deadline': 1})
assert s == 200 and r['category'] in ('Homework', 'Exam', 'Quiz', 'Project'), (s, r)
assert r['importance'] in ('high', 'medium', 'low') and 0.5 <= r['time_est'] <= 8, r
print('predict   ok ', r)

s, t = call('POST', '/api/tasks', {'task_text': 'finish bio hw', 'deadline': 1, **{k: r[k] for k in ('category', 'importance', 'time_est')}})
assert s == 201 and 'id' in t, (s, t)
print('add task  ok ', t)

s, r = call('PATCH', f"/api/tasks/{t['id']}", {'status': 'done'})
assert s == 200, (s, r)
s, tasks = call('GET', '/api/tasks')
assert s == 200 and any(x['id'] == t['id'] and x['status'] == 'done' for x in tasks), tasks
print('tick task ok ', tasks[0])

s, r = call('DELETE', f"/api/tasks/{t['id']}")
assert s == 200, (s, r)
s, r = call('POST', '/api/logout')
assert s == 200, (s, r)
s, _ = call('GET', '/api/me')
assert s == 401, 'still logged in after logout'
print('cleanup   ok')

print(f'\nALL GOOD — {BASE} is working.')
