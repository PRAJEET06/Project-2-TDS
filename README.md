# Quiz Solver (LLM-assisted automated quiz solver)

## What this project does
This service accepts POST requests at `/solve` with JSON containing `email`, `secret`, and `url`. It visits the URL (headless Playwright), extracts instructions/files, computes an answer with heuristics (CSV/PDF parsing), and submits it to the page-specified submit endpoint. It supports chaining: when the submit response returns another URL, the service follows it until completion or a 3-minute timeout.

## How to run locally
1. Create and activate a Python venv.
2. `pip install -r requirements.txt`
3. `python -m playwright install`
4. `set QUIZ_SECRET=my_secret_here` (or create `.env` from `.env.example`)
5. `python app.py`
6. Test with `python test_request.py`

## Files
- `app.py` — server implementation
- `test_request.py` — sample test client
- `RUN_INSTRUCTIONS.md` — run & test steps

## License
MIT
