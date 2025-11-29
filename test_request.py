# test_request.py
import requests, json, sys
payload = {"email":"you@example.com","secret":"my_local_secret_123","url":"https://tds-llm-analysis.s-anand.net/demo"}
try:
    r = requests.post("http://127.0.0.1:8000/solve", json=payload, timeout=60)
except Exception as e:
    print("REQUEST ERROR:", e)
    sys.exit(1)

print("HTTP", r.status_code)
text = r.text
# print first 2000 chars so it doesn't flood terminal
print(text[:2000])
