# app.py  -- Part 4: chaining, 3-minute window, last-submission-wins
import os
import sys
import asyncio
import re
import base64
import io
import time
from typing import Optional, Any, Dict, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import pdfplumber
import httpx
from playwright.async_api import async_playwright

# Ensure Windows uses proactor loop for subprocess support
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

SECRET = os.getenv("QUIZ_SECRET", "my_local_secret_123")
USER_AGENT = "quiz-solver/1.0 (+https://example)"

app = FastAPI(title="Quiz Solver - Part 4 (chaining)")

# regexes & helpers (same as before)
ATOB_RE = re.compile(r"atob\(\s*(?:`([^`]+)`|'([^']+)'|\"([^\"]+)\")\s*\)", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
FILE_RE_EXT = re.compile(r".*\.(pdf|csv|xlsx|xls)$", re.IGNORECASE)
PAGE_NUM_RE = re.compile(r"page\s+(\d+)", re.IGNORECASE)
VALUE_COL_CANDIDATES = ["value", "Value", "amount", "Amount", "total", "Total"]


def decode_atob_from_html(html: str) -> Optional[str]:
    m = ATOB_RE.search(html)
    if not m:
        return None
    b64 = next(g for g in m.groups() if g)
    try:
        return base64.b64decode(b64).decode("utf-8", errors="replace")
    except Exception:
        return None


def find_urls_in_text(text: str) -> List[str]:
    return URL_RE.findall(text or "")


def pick_submit_url_from_texts(page_text: str, all_urls: List[str]) -> Optional[str]:
    opts = URL_RE.findall(page_text or "")
    for u in opts:
        if "/submit" in u.lower():
            return u
    for u in all_urls:
        if "/submit" in u.lower() or "submit" in u.lower():
            return u
    if opts:
        return opts[0]
    return all_urls[0] if all_urls else None


def choose_files_from_urls(urls: List[str]) -> List[str]:
    return [u for u in urls if FILE_RE_EXT.match(u)]


async def download_url_to_bytes(url: str, timeout: int = 30) -> Optional[bytes]:
    try:
        headers = {"User-Agent": USER_AGENT}
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            return r.content
    except Exception:
        return None


def parse_table_from_pdf_bytes(pdf_bytes: bytes, page_hint: Optional[int] = None) -> List[pd.DataFrame]:
    dfs = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = pdf.pages
            page_indexes = range(len(pages)) if page_hint is None else [page_hint - 1]
            for pi in page_indexes:
                if pi < 0 or pi >= len(pages):
                    continue
                page = pages[pi]
                try:
                    tables = page.extract_tables()
                    if tables:
                        for tbl in tables:
                            if len(tbl) >= 2:
                                df = pd.DataFrame(tbl[1:], columns=tbl[0])
                            else:
                                df = pd.DataFrame(tbl)
                            dfs.append(df)
                except Exception:
                    pass
    except Exception:
        pass
    return dfs


def parse_csv_or_excel_bytes(bytes_data: bytes, filename: str) -> Optional[pd.DataFrame]:
    try:
        if filename.lower().endswith(".csv"):
            return pd.read_csv(io.BytesIO(bytes_data))
        else:
            return pd.read_excel(io.BytesIO(bytes_data), engine="openpyxl")
    except Exception:
        return None


def find_value_column_and_aggregate(df: pd.DataFrame) -> Optional[float]:
    for cand in VALUE_COL_CANDIDATES:
        for c in df.columns:
            if str(c).strip().lower() == cand.lower():
                try:
                    nums = pd.to_numeric(df[c], errors="coerce")
                    total = float(nums.sum(skipna=True))
                    return total
                except Exception:
                    continue
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        best = max(numeric_cols, key=lambda c: df[c].notnull().sum())
        try:
            total = float(df[best].sum(skipna=True))
            return total
        except Exception:
            return None
    return None


def heuristic_extract_page_hint(text: str) -> Optional[int]:
    m = PAGE_NUM_RE.search(text or "")
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def build_submit_payload(email: str, secret: str, url: str, answer: Any) -> dict:
    return {"email": email, "secret": secret, "url": url, "answer": answer}


# core single-URL handler: fetch, try heuristics, compute answer (non-blocking async)
async def handle_one(quiz_url: str, email: str, secret: str, time_left: float) -> Dict[str, Any]:
    """
    Return dict: { 'visited_url', 'computed_answer' (or None), 'submit_url', 'submit_response', 'reasoning' }
    """
    start = time.time()
    reasoning = []
    computed_answer = None
    submit_response = None
    submit_url = None
    file_links = []

    # Step A: fetch page via async playwright
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(quiz_url, timeout=min(60000, int(time_left * 1000)))
            title = await page.title()
            page_html = await page.content()
            try:
                elem = await page.query_selector("#result")
                page_text = await (await elem.inner_text() if elem else await page.inner_text("body"))
            except Exception:
                page_text = await page.inner_text("body")
            await browser.close()
    except Exception as e:
        reasoning.append(f"playwright_error: {e}")
        return {"visited_url": quiz_url, "computed_answer": None, "submit_url": None, "submit_response": None, "reasoning": reasoning}

    decoded = decode_atob_from_html(page_html)
    decoded_preview = decoded[:200] + "..." if decoded and len(decoded) > 200 else decoded

    # gather urls
    urls = list(dict.fromkeys(find_urls_in_text(page_html) + find_urls_in_text(decoded or "")))
    urls += [u for u in find_urls_in_text(page_text) if u not in urls]

    submit_url = pick_submit_url_from_texts(page_text, urls)
    if "/submit" not in (submit_url or ""):
        m = URL_RE.search(page_text or "")
        if m and "/submit" in m.group(0):
            submit_url = m.group(0)

    file_links = choose_files_from_urls(urls)

    # Heuristic inline answers
    # numeric
    inline_num = re.search(r"\"answer\"\s*:\s*([0-9]+)", page_text)
    if inline_num:
        computed_answer = int(inline_num.group(1))
        reasoning.append("found inline numeric answer in page text")
    else:
        # string example
        inline_str = re.search(r"\"answer\"\s*:\s*\"([^\"]*)\"", page_text)
        if inline_str:
            example = inline_str.group(1).strip()
            if example == "" or "anything" in example.lower() or "what you want" in example.lower():
                computed_answer = "demo-answer-123"
                reasoning.append("page requested any answer; using default demo string")
            else:
                mnum = re.search(r"[-+]?\d*\.\d+|\d+", example.replace(",", ""))
                if mnum:
                    try:
                        computed_answer = int(mnum.group(0))
                        reasoning.append("found numeric inside string example")
                    except Exception:
                        computed_answer = example
                        reasoning.append("using example string as answer")
                else:
                    computed_answer = example
                    reasoning.append("using example string as answer")

    page_hint = heuristic_extract_page_hint(page_text)
    # Try file links
    for furl in file_links:
        if time.time() - start > time_left - 2:
            reasoning.append("timeout before finishing downloads on this url")
            break
        b = await download_url_to_bytes(furl)
        if not b:
            reasoning.append(f"failed to download {furl}")
            continue
        if furl.lower().endswith(".pdf"):
            dfs = parse_table_from_pdf_bytes(b, page_hint=page_hint)
            if dfs:
                for df in dfs:
                    val = find_value_column_and_aggregate(df)
                    if val is not None:
                        computed_answer = val
                        reasoning.append(f"found value column in PDF table from {furl}")
                        break
            if computed_answer is None and page_hint:
                try:
                    with pdfplumber.open(io.BytesIO(b)) as pdf:
                        if 0 <= (page_hint - 1) < len(pdf.pages):
                            txt = pdf.pages[page_hint - 1].extract_text() or ""
                            nums = re.findall(r"[-+]?\d*\.\d+|\d+", txt.replace(",", ""))
                            nums = [float(n) for n in nums]
                            if nums:
                                computed_answer = sum(nums)
                                reasoning.append(f"sum of numeric tokens on PDF page {page_hint}")
                except Exception:
                    pass
        else:
            df = parse_csv_or_excel_bytes(b, furl)
            if df is not None:
                val = find_value_column_and_aggregate(df)
                if val is not None:
                    computed_answer = val
                    reasoning.append(f"found value column in {furl}")
                    break

    # decoded JSON fallback
    if computed_answer is None and decoded:
        try:
            import json as _json
            obj = _json.loads(decoded)
            if isinstance(obj, dict) and "answer" in obj and isinstance(obj["answer"], (int, float)):
                computed_answer = obj["answer"]
                reasoning.append("found answer field inside decoded JSON")
        except Exception:
            pass

    # If still None and submit_url exists & no files -> fallback 42
    if computed_answer is None and submit_url and not file_links:
        computed_answer = 42
        reasoning.append("no files & no inline answer; using fallback 42 to submit")

    # If still None -> return partial result
    if computed_answer is None:
        return {
            "visited_url": quiz_url,
            "computed_answer": None,
            "submit_url": submit_url,
            "submit_response": None,
            "reasoning": reasoning,
        }

    # If there's no submit_url (page just wants you to scrape a code), do NOT attempt to POST.
    # Return the computed answer so the orchestrator can stop the chain gracefully.
    if not submit_url:
        return {
            "visited_url": quiz_url,
            "computed_answer": computed_answer,
            "submit_url": None,
            "submit_response": None,
            "reasoning": reasoning,
        }

    # Build payload and submit
    payload = build_submit_payload(email=email, secret=secret, url=quiz_url, answer=computed_answer)

    # normalize submit_url
    if submit_url and re.match(r"^https?://[^/]+/?$", submit_url):
        submit_url = submit_url.rstrip("/") + "/submit"

    try:
        headers = {"User-Agent": USER_AGENT, "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(submit_url, json=payload, headers=headers)
            r.raise_for_status()
            try:
                submit_response = r.json()
            except Exception:
                submit_response = {"raw_text": r.text}
    except Exception as e:
        reasoning.append(f"submit_failed: {e}")
        return {
            "visited_url": quiz_url,
            "computed_answer": computed_answer,
            "submit_url": submit_url,
            "submit_response": None,
            "reasoning": reasoning,
        }

    return {
        "visited_url": quiz_url,
        "computed_answer": computed_answer,
        "submit_url": submit_url,
        "submit_response": submit_response,
        "reasoning": reasoning,
    }


@app.post("/solve")
async def solve(request: Request):
    # top-level orchestrator: follow chain until no next url or time runs out
    overall_start = time.time()
    MAX_SECONDS = 170  # safety cutoff (< 180s / 3 minutes)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    email = payload.get("email")
    secret = payload.get("secret")
    start_url = payload.get("url")

    if secret != SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    if not start_url:
        raise HTTPException(status_code=400, detail="Missing 'url' in payload")

    history = []
    current_url = start_url
    time_left = MAX_SECONDS

    while True:
        elapsed = time.time() - overall_start
        time_left = MAX_SECONDS - elapsed
        if time_left < 5:
            # not enough time to safely process another URL
            return {"status": "timed_out", "history": history, "message": "ran out of time before finishing chain"}

        step = await handle_one(current_url, email, secret, time_left)
        history.append(step)

        # If we didn't compute an answer for this URL, stop chaining and return partial
        if step.get("computed_answer") is None:
            return {"status": "partial", "history": history, "message": "could not compute answer for a URL in chain"}

        # Check submit response for next URL
        submit_resp = step.get("submit_response") or {}
        # The demo uses field "url" in submit_response - adapt if different
        next_url = submit_resp.get("url") if isinstance(submit_resp, dict) else None
        if not next_url:
            # finished
            return {"status": "finished", "history": history}
        # avoid infinite loops: break if next_url repeats
        if any(h.get("visited_url") == next_url for h in history):
            return {"status": "finished", "history": history, "message": "next url repeated; stopping to avoid loop"}
        # continue to next_url
        current_url = next_url

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
